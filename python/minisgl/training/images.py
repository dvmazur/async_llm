import torch


def inputs_embeds(model, replay):
    """Re-encode recorded images and scatter them into packed token embeddings."""
    states = model.get_input_embeddings()(replay.input_ids)
    if replay.trace is None:
        raise ValueError("Image replay requires its original trace")
    image_id = getattr(model.config, "image_token_id", None)
    inverse = replay.order.argsort()
    indices, features, pixels, grids, image_rows = [], [], [], [], []
    start = 0
    for batch in replay.trace.batches:
        for request in batch.requests:
            ids = torch.tensor(request.input_ids)
            types = request.mm_token_type_ids
            if types is not None:
                types = types.tensor().flatten()
                if len(types) != len(ids) or not torch.all((types == 0) | (types == 1)):
                    raise ValueError(
                        "Image token types must align with inputs and contain only 0/1"
                    )
                mask = types == 1
                if image_id is not None and not torch.equal(mask, ids == image_id):
                    raise ValueError("Image token types disagree with the model's image token ID")
            else:
                mask = ids == image_id if image_id is not None else torch.zeros_like(ids).bool()
            has_pixels = request.pixel_values is not None
            if not has_pixels and request.image_embeds is None:
                if request.image_grid_thw is not None or mask.any():
                    raise ValueError("Image tokens require recorded pixels or image embeddings")
                start += len(ids)
                continue
            count = int(mask.sum())
            if not count and has_pixels:
                raise ValueError("Recorded image features require matching image tokens")
            rows = inverse[torch.arange(start, start + len(ids))[mask].to(inverse.device)]
            if has_pixels:
                config = getattr(model.config, "vision_config", None)
                if config is None or not hasattr(model, "get_image_features"):
                    raise ValueError("Raw image replay requires a conditional-generation model")
                if request.image_grid_thw is None:
                    raise ValueError("Recorded pixels require image_grid_thw")
                grid = request.image_grid_thw.tensor()
                values = request.pixel_values.tensor()
                merge = config.spatial_merge_size
                if (
                    grid.ndim != 2
                    or grid.shape[1] != 3
                    or grid.dtype not in (torch.int32, torch.int64)
                    or not len(grid)
                    or not torch.all(grid > 0)
                    or not torch.all(grid[:, 1:] % merge == 0)
                ):
                    raise ValueError("Image grids must contain positive, merge-aligned [t, h, w]")
                patches = int(grid.prod(-1).sum())
                width = config.in_channels * config.temporal_patch_size * config.patch_size**2
                if values.shape != (patches, width) or not values.is_floating_point():
                    raise ValueError("Recorded pixels do not match the image grid and patch shape")
                if patches // merge**2 != count:
                    raise ValueError("Image grid feature count does not match image tokens")
                pixels.append(values)
                grids.append(grid)
                image_rows.append(rows)
            else:
                values = request.image_embeds.tensor()
                if values.shape != (count, states.shape[-1]) or not values.is_floating_point():
                    raise ValueError("Recorded image embeddings do not match image tokens")
                indices.append(rows)
                features.append(values.to(device=states.device, dtype=states.dtype))
            start += len(ids)
    if pixels:
        output = model.get_image_features(
            torch.cat(pixels).to(states.device),
            torch.cat(grids).to(states.device),
            return_dict=True,
        ).pooler_output
        output = torch.cat(output) if isinstance(output, (tuple, list)) else output
        rows = torch.cat(image_rows)
        if output.shape != (len(rows), states.shape[-1]):
            raise ValueError("Vision output does not match recorded image tokens")
        indices.append(rows)
        features.append(output.to(device=states.device, dtype=states.dtype))
    return states.index_copy(0, torch.cat(indices), torch.cat(features)) if indices else states
