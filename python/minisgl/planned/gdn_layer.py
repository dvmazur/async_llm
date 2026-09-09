"""A complete capacity GDN attention sublayer, sharing scratch across layers.

This is not the full model runner. One profile owns GDNWorkspace and its two
phase recipes. Layer bindings contain weights only; block lookup/upload/commit
must already be complete. Dense and serialized block-FP8 projections share
fixed-output Linear bindings; unsupported backends select compatibility before writes.
"""
import torch
import triton
from types import SimpleNamespace

from .conv_device import BoundConv
from .gdn_device import BoundGDN, DevicePhase, DeviceRows, _DeviceTables
from . import gdn_layer_kernels as kernels
from .linear import BoundLinear,LinearWorkspace


class _NormRecipes(_DeviceTables):
    fields=("recipes",)


def _norm_recipes(plan,heads,sms):
    sizes=(sum(r.length for r in plan.prefill_requests),len(plan.decode_requests))
    return SimpleNamespace(recipes=tuple(min(4,triton.next_power_of_2(max(1,triton.cdiv(n*heads,2*sms)))) for n in sizes))


class GDNWorkspace:
    def __init__(self,pool,plan,*,key_heads,linear_workspace=None,rows=None):
        cap=plan.capacity
        self.pool,self.capacity,self.key_heads=pool,cap,key_heads
        self.linear_workspace=linear_workspace
        self.owns_rows=rows is None
        self.rows=DeviceRows(plan.rows,pool.affine.device) if rows is None else rows
        self.pf_phase=DevicePhase(plan.prefill,pool.affine.device)
        self.dec_phase=DevicePhase(plan.decode,pool.affine.device)
        s=pool.shape
        self.sms=torch.cuda.get_device_properties(pool.affine.device).multi_processor_count
        self.norm_recipes=_NormRecipes(_norm_recipes(plan,s.heads,self.sms),pool.affine.device)
        if (not key_heads or s.heads%key_heads or s.conv_channels!=(2*key_heads+s.heads)*s.dim
                or s.slots!=cap.block_slots or not pool.affine.is_cuda
                or bool(cap.prefill_requests)!=bool(cap.prefill_tokens)):
            raise ValueError("incompatible GDN pool/profile")
        p,d=cap.prefill_tokens,cap.decode_workers
        self.total=p+d
        if not self.total:
            raise ValueError("empty profile")
        dtype,device=pool.conv.dtype,pool.conv.device
        def buf(*shape,dtype=dtype):return torch.empty(shape,device=device,dtype=dtype)
        self.raw=buf(self.total,s.conv_channels)
        self.convolved=buf(self.total,s.conv_channels)
        self.z=buf(self.total,s.heads*s.dim)
        self.a,self.b=buf(self.total,s.heads),buf(self.total,s.heads)
        self.g=buf(self.total,s.heads,dtype=torch.float32)
        self.beta_pf=buf(p,s.heads)
        self.beta_dec=buf(d,s.heads,dtype=torch.float32)
        self.alpha_dec=buf(d,s.heads,dtype=torch.float32)
        self.core=buf(self.total,s.heads,s.dim)
        self.normed=buf(self.total,s.heads*s.dim)
        # PF completes before D: frontier/initial/windows can share one arena.
        # Flat slices yield contiguous phase-specific layouts, without copying.
        width=max(cap.prefill_requests,d)
        self.compose_storage=buf(3*width*s.heads*s.dim*s.dim,dtype=torch.float32)
        self.window_storage=buf(width*s.conv_channels*s.conv_window)
        def compose(phase):
            n=phase.width
            workspace=self.compose_storage[:3*n*s.heads*s.dim*s.dim].view(3,n,s.heads,s.dim,s.dim)
            return BoundGDN(pool.affine,phase,workspace=workspace)
        self.pf_compose=compose(self.pf_phase) if p else None
        self.dec_compose=compose(self.dec_phase) if d else None
        self.pf_window=self.window_storage[:cap.prefill_requests*s.conv_channels*s.conv_window].view(
            cap.prefill_requests,s.conv_channels,s.conv_window)
        self.dec_window=self.window_storage[:d*s.conv_channels*s.conv_window].view(d,s.conv_channels,s.conv_window)
        q,k,v=self.convolved.split([key_heads*s.dim,key_heads*s.dim,s.heads*s.dim],-1)
        self.pf_q=q[:p].view(1,p,key_heads,s.dim)
        self.pf_k=k[:p].view(1,p,key_heads,s.dim)
        self.pf_v=v[:p].view(1,p,s.heads,s.dim)
        self.dec_q=q[p:].view(d,1,key_heads,s.dim)
        self.dec_k=k[p:].view(d,1,key_heads,s.dim)
        self.dec_v=v[p:].view(d,1,s.heads,s.dim)
        self.pf_g=self.g[:p].unsqueeze(0)
        self.dec_g=self.g[p:].unsqueeze(1)
        self.pf_beta=self.beta_pf.unsqueeze(0)
        self.dec_beta=self.beta_dec.unsqueeze(1)
        self.dec_alpha=self.alpha_dec.unsqueeze(1)
        self.pf_core=self.core[:p].unsqueeze(0)
        self.dec_core=self.core[p:].unsqueeze(1)
        self.fla=None
        if p:
            from .fla_device import BoundFLA
            self.fla=BoundFLA(pool.affine,self.pf_phase,self.rows,cap,key_heads=key_heads,dtype=dtype,
                              old_layer_rounding=True)

    def upload(self,plan):
        if plan.capacity!=self.capacity:
            raise ValueError("plan requires a different profile")
        if self.owns_rows:self.rows.upload(plan.rows)
        self.pf_phase.upload(plan.prefill)
        self.dec_phase.upload(plan.decode)
        self.norm_recipes.upload(_norm_recipes(plan,self.pool.shape.heads,self.sms))


class BoundGDNLayer:
    def __init__(self,layer,workspace:GDNWorkspace):
        self.layer,self.ws=layer,workspace
        s=workspace.pool.shape
        if (layer.num_k_heads!=workspace.key_heads or layer.num_v_heads!=s.heads
                or layer.head_k_dim!=s.dim or layer.head_v_dim!=s.dim
                or layer.conv_kernel!=s.conv_window or not 0<=layer._lin_idx<s.layers):
            raise ValueError("layer incompatible with GDN workspace")
        self.projections=(layer.in_proj_qkv,layer.in_proj_z,layer.in_proj_a,layer.in_proj_b)
        self.projection_outputs=(workspace.raw,workspace.z,workspace.a,workspace.b)
        projections=(*self.projections,layer.out_proj)
        if any(p.weight_scale_inv is not None for p in projections) and workspace.linear_workspace is None:
            workspace.linear_workspace=LinearWorkspace(workspace.total,
                max(p.full_input_size for p in projections),device=workspace.raw.device)
        self.inputs=tuple(BoundLinear(p,workspace.total,dtype=workspace.raw.dtype,
            workspace=workspace.linear_workspace,active=workspace.rows.active) for p in self.projections)
        self.output=BoundLinear(layer.out_proj,workspace.total,dtype=workspace.raw.dtype,
            workspace=workspace.linear_workspace,active=workspace.rows.active)
        self.pf_conv=self.dec_conv=None
        if workspace.capacity.prefill_tokens:
            self.pf_conv=BoundConv(workspace.pool.conv,layer.conv1d.weight,workspace.pf_phase,
                workspace.rows,workspace.capacity,prefill=True,window_workspace=workspace.pf_window)
        if workspace.capacity.decode_workers:
            self.dec_conv=BoundConv(workspace.pool.conv,layer.conv1d.weight,workspace.dec_phase,
                workspace.rows,workspace.capacity,prefill=False,window_workspace=workspace.dec_window)

    def run(self,x,output):
        w,l=self.ws,self.layer
        if (x.shape!=(w.total,l.in_proj_qkv.full_input_size)
                or output.shape!=(w.total,l.out_proj.full_output_size)
                or x.device!=w.raw.device or output.device!=w.raw.device
                or x.dtype!=w.raw.dtype or output.dtype!=w.raw.dtype
                or not x.is_contiguous() or not output.is_contiguous()):
            raise ValueError("invalid GDN layer input/output")
        for projection,out in zip(self.inputs,self.projection_outputs):
            projection.run(x,out)
        cap=w.capacity
        kernels.gates[(triton.cdiv(w.total*l.num_v_heads,256),)](
            w.a,w.b,l.A_log,l.dt_bias,w.rows.active,w.g,w.beta_pf,w.beta_dec,w.alpha_dec,
            P=cap.prefill_tokens,D=cap.decode_workers,H=l.num_v_heads,BLOCK=256,
            enable_fp_fusion=False)
        if self.pf_conv is not None:
            self.pf_conv.run(l._lin_idx,w.raw,w.convolved)
            w.pf_compose.compose(l._lin_idx)
            w.fla.run(l._lin_idx,w.pf_q,w.pf_k,w.pf_v,w.pf_g,w.pf_beta,
                      w.pf_compose.initial,w.pf_core,conv=self.pf_conv)
        # Neither compose nor conv for decode may move above PF publication.
        if self.dec_conv is not None:
            self.dec_conv.run(l._lin_idx,w.raw,w.convolved)
            w.dec_compose.compose(l._lin_idx)
            w.dec_compose.decode(l._lin_idx,w.dec_q,w.dec_k,w.dec_v,w.dec_g,w.dec_beta,
                                 w.dec_alpha,w.dec_core,conv=self.dec_conv)
        kernels.gated_norm[(triton.cdiv(w.total*l.num_v_heads,4),)](
            w.core,w.z,l.norm.weight,w.rows.active,w.normed,R=w.total,H=l.num_v_heads,
            D=l.head_v_dim,EPS=l.norm._eps,BD=triton.next_power_of_2(l.head_v_dim),
            ROWS=4,recipes=w.norm_recipes.recipes,P=cap.prefill_tokens,
            num_warps=min(max(triton.next_power_of_2(l.head_v_dim)//256,1),8))
        return self.output.run(w.normed,output)


class PlannedGDNLayers:
    graph_compatible=True

    def __init__(self,layers,pool,plan,*,rows=None,linear_workspace=None):
        self.workspace=GDNWorkspace(pool,plan,key_heads=layers[0].num_k_heads,
                                    rows=rows,linear_workspace=linear_workspace)
        self.layers=[BoundGDNLayer(layer,self.workspace) for layer in layers]

    def upload(self,plan):self.workspace.upload(plan)

    def run(self,index,x,output):return self.layers[index].run(x,output)


def bind_gdn_layers(layers,pool,plan,*,rows=None,linear_workspace=None):
    """Preflight the layer set, never switch after a layer has written states.

    The eventual full decoder runner must respect graph_compatible for *all*
    consumers and select eager for the whole forward if one cannot capture.
    """
    from minisgl.models import qwen3_5_delta as delta
    if not layers:raise ValueError("at least one GDN layer required")
    reason=None
    if delta._fla_chunk is None or delta._fla_recurrent is None:
        reason="FLA unavailable; preserve old no-FLA normalization and scan"
    elif pool.shape.dim>256:
        reason="dimension outside bounded FLA recipe"
    elif torch.cuda.get_device_capability(pool.affine.device)<(8,9) and any(
            p.weight_scale_inv is not None for l in layers for p in (
                l.in_proj_qkv,l.in_proj_z,l.in_proj_a,l.in_proj_b,l.out_proj)):
        reason="native FP8 unavailable on this device"
    if reason is not None:
        from .gdn_fallback import EagerGDNLayers
        return EagerGDNLayers(layers,pool,plan,reason)
    return PlannedGDNLayers(layers,pool,plan,rows=rows,linear_workspace=linear_workspace)
