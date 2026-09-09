"""Generic full-body capture plus outer prepare/retain/completion protocol."""
from dataclasses import dataclass
import torch

from .forward_plan import prepare_forward
from .attention_plan import prepare_attention
from .model_io import prepare_inputs


class CapturedBody:
    """Does not know about layers, blocks, GDN, KV, request IDs or topology."""
    def __init__(self,body):
        self.body=body
        self.graph=None
        self.result=None
        self.captures=self.replays=0

    @torch.inference_mode()
    def capture(self):
        if self.graph is not None:raise RuntimeError("body already captured")
        self.body()  # caller supplies a side-effect-free warmup transaction
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):self.result=self.body()
        self.graph=graph
        self.captures+=1

    def replay(self):
        if self.graph is None:raise RuntimeError("body must be captured before replay")
        self.graph.replay()
        self.replays+=1
        return self.result


@dataclass(frozen=True)
class ExecutionResult:
    logits:torch.Tensor  # owned snapshot, never the graph's reusable output
    completion:torch.cuda.Event
    used_graph:bool


class ProgramRunner:
    """One profile, one in-flight forward; outer session owns block commit.

    prepare_execution may warm/capture *only* an all-inactive plan, which reads
    no live GDN/KV and writes no persistent states. execute_prepared is the
    first method that can write live model state; mark a transaction submitted
    before calling it, and commit only after returned completion is ready.
    """
    def __init__(self,program,*,use_graph=True):
        self.program=program
        self.use_graph=use_graph and program.graph_compatible
        self.body=CapturedBody(program.run) if self.use_graph else None
        self.prepared=False
        self._inflight=None
        self.forward_count=self.eager_count=0
        self.failed_count=0
        self.poisoned=False
        self._retain_indices=None

    def _available(self):
        if self.poisoned:raise RuntimeError("decoder runner completion failed; rebuild runtime")
        try:ready=self._inflight is None or self._inflight.query()
        except Exception:
            self.poisoned=True
            raise
        if not ready:raise RuntimeError("previous decoder execution is still in flight")

    @torch.inference_mode()
    def prepare_execution(self,forward,attention,inputs,features=None):
        self._available()
        self.prepared=False
        p=self.program
        if self.body is not None and self.body.graph is None:
            empty=prepare_forward({},capacity=p.capacity)
            empty_attention=prepare_attention({},empty,attention.capacity)
            empty_inputs=prepare_inputs(empty,[],vocab_size=p.vocab)
            p.prepare(empty,empty_attention,empty_inputs)
            self.body.capture()
        p.prepare(forward,attention,inputs,features)
        valid=[i for i,row in enumerate(forward.rows.output_rows) if row>=0]
        self._retain_indices=torch.tensor(valid,device=p.device,dtype=torch.int64)
        self.prepared=True

    @torch.inference_mode()
    def execute_prepared(self):
        self._available()
        if not self.prepared:raise RuntimeError("execution was not prepared or was already consumed")
        self.prepared=False
        # A caller marks the transaction submitted BEFORE entering this method.
        # On failure it must invalidate potential writes and record a drain event.
        try:
            raw=self.body.replay() if self.body is not None else self.program.run()
            snapshot=raw.index_select(0,self._retain_indices)
            event=torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.program.device))
        except Exception:
            self.failed_count+=1
            try:
                self._inflight=torch.cuda.Event()
                self._inflight.record(torch.cuda.current_stream(self.program.device))
            except Exception:self.poisoned=True
            raise
        self._inflight=event
        self.forward_count+=1
        self.eager_count+=self.body is None
        return ExecutionResult(snapshot,event,self.body is not None)
