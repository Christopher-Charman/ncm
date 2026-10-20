"""Small inference primitives that can be tested without evaluation CLI side effects."""


def decode_final_logits(vqvae, hierarchical_codes):
    """Decode only the final latent from a complete hierarchy of VQ indices.

    This intentionally does not allocate or return intermediate reconstructions.
    Use vqvae.reconstruct_multiscales when progressive stages are actually needed.
    """
    final_latent = vqvae.vq.idx_to_z_q(hierarchical_codes)
    return vqvae.decoder(final_latent)
