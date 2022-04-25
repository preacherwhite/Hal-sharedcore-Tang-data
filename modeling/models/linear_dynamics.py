import torch
import torch.nn as nn

class LinearLatent(nn.Module):
    """
    A linear latent-space dynamical systems model.

    Has a matrix (and its transpose) to transform between
    the (neural) data and the latent space, then a transition
    matrix for the latent space.
    """
    def __init__(self, n_in, n_latent, n_steps):
        """
        n_in is the dimension of the data to be modeled

        n_latent is the dimensionality of the latent space
        (probably lower than n_in)
        
        n_steps is the default number of time steps
        to predict for
        """
        super().__init__()

        self.n_in = n_in
        self.n_latent = n_latent
        self.n_steps = n_steps

        self.transform = nn.Parameter(
                torch.empty(n_in, n_latent))
        self.transition = nn.Parameter(
                torch.empty(n_latent, n_latent))

        self.initialize_weights()

    def initialize_weights(self):
        # not sure about these
        nn.init.normal_(self.transform, std=0.1)
        nn.init.normal_(self.transition, std=0.1)

    def forward(self, x, steps=None, latent=False):
        """
        x is the initial state

        steps is the number of additional time steps
        to run for (overriding the default)

        returns an (n_in X steps) tensor, not including the input
        (if latent is set to true, also returns the 
        (n_latent X steps+1) full latent space tensor)
        """
        n_batch = x.shape[0]
        steps = self.n_steps if steps is None else steps
        out = torch.empty(n_batch, self.n_in, steps)
        latent_state = x.matmul(self.transform)

        if latent:
            lat_out = torch.empty(n_batch, self.n_latent, steps+1)
            lat_out[...,0] = latent_state

        for i in range(steps):
            latent_state = latent_state.matmul(self.transition)

            out[...,i] = latent_state.matmul(self.transform.transpose(1,0))

            if latent:
                lat_out[...,i+1] = latent_state

        if latent:
            return out, lat_out

        return out
