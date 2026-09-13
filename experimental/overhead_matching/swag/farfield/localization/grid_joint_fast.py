"""Opt-in fused FP32 joint likelihood; same global candidates and grid."""
import math
import torch


def epoch_term(log_prod, east, north, cand_east, cand_north, extent, p):
    d_east = cand_east[None, :] - (east + p[0])[:, None]
    d_north = cand_north[None, :] - (north + p[1])[:, None]
    distance = torch.sqrt(d_east * d_east + d_north * d_north)
    safe_distance = torch.clamp(distance, min=1.0)
    quant = p[14] * (p[7] + (p[8] / safe_distance) ** 2)
    variance = (p[3] + (p[6] / safe_distance) ** 2 + quant + p[4]
                + (p[5] / safe_distance) ** 2)
    variance = variance + (extent[None, :] / safe_distance) ** 2
    kappa = 1.0 / variance
    cos_delta = torch.cos(torch.atan2(d_east, d_north) - p[2])
    term = -torch.log(torch.special.i0e(kappa)) + kappa * (cos_delta - 1.0)
    excess = torch.clamp(distance - p[10], min=0.0)
    soft = torch.exp(-0.5 * (excess / (p[11] * p[10])) ** 2)
    term = term + p[15] * torch.log((1.0 - p[12]) * soft + p[12] + 1e-30)
    return log_prod + p[9] * term


_compiled_epoch = None


def joint_likelihood(belief, epochs, cand_east, cand_north, cand_weight,
                     sigma_pos, pi0, tail_mass, *, quantization_comp=True,
                     chunk=64, range_softness=0.25, range_floor=0.0,
                     cand_extent=None, temper=1.0, mixture="sum", cap=None,
                     compiled=True):
    global _compiled_epoch
    if compiled and _compiled_epoch is None:
        _compiled_epoch = torch.compile(epoch_term, dynamic=True, fullgraph=True)
    kernel = _compiled_epoch if compiled else epoch_term
    n_cells = belief.cell_east.numel()
    mix = torch.zeros((belief.n_heading, n_cells), device=belief.device)
    binw = 2 * math.pi / belief.n_heading
    extent = torch.zeros_like(cand_east) if cand_extent is None else cand_extent
    # Scalars are tensor inputs to avoid a compilation for every observation.
    parameters = []
    for h in belief.bin_rad:
        parameters.append([
            [df * math.sin(h) - dl * math.cos(h),
             df * math.cos(h) + dl * math.sin(h), h + dh + obs,
             base, hs, ps, sigma_pos, binw * binw / 12,
             belief.grid.cell_m / math.sqrt(12), temper,
             1.0 if rm is None else rm, range_softness, range_floor,
             0.0, float(quantization_comp), float(rm is not None)]
            for df, dl, dh, obs, base, rm, hs, ps in epochs])
    params = torch.tensor(parameters, dtype=torch.float32, device=belief.device)
    for heading in range(belief.n_heading):
        for start in range(0, cand_east.numel(), chunk):
            sl = slice(start, min(start + chunk, cand_east.numel()))
            log_prod = torch.zeros((n_cells, sl.stop - sl.start), device=belief.device)
            for i in range(len(epochs)):
                log_prod = kernel(log_prod, belief.cell_east, belief.cell_north,
                                  cand_east[sl], cand_north[sl], extent[sl],
                                  params[heading, i])
            contrib = torch.exp(torch.clamp(log_prod, max=80.0)) * cand_weight[sl][None, :]
            if mixture == "max":
                mix[heading] = torch.maximum(mix[heading], contrib.max(dim=1).values)
            else:
                mix[heading] += contrib.sum(dim=1)
    if cap is not None:
        mix = torch.clamp(mix, max=cap)
    return (pi0 + (1 - pi0) * (mix + tail_mass)).view(
        belief.n_heading, belief.grid.n_north, belief.grid.n_east)
