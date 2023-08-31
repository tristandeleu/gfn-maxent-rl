import jax.numpy as jnp
import jraph
import haiku as hk

from collections import namedtuple


Features = namedtuple('Features', ['nodes', 'globals'])


class GNNBackbone(hk.Module):
    def __init__(self, num_layers=1, name=None):
        super().__init__(name=name)
        self.num_layers = num_layers

    def __call__(self, graphs, masks):
        batch_size, num_variables = masks.shape[:2]

        # Embedding of the nodes & edges
        node_embeddings = hk.Embed(num_variables, embed_dim=128)
        edge_embeddings = hk.get_parameter(
            'edge_embed',
            shape=(1, 128),
            init=hk.initializers.TruncatedNormal()
        )
        global_embeddings = hk.get_parameter(
            'global_embed',
            shape=(1, 128),
            init=hk.initializers.TruncatedNormal()
        )

        features = graphs._replace(
            nodes=node_embeddings(graphs.nodes),
            edges=jnp.repeat(edge_embeddings, graphs.edges.shape[0], axis=0),
            globals=jnp.repeat(global_embeddings, graphs.n_node.shape[0], axis=0),
        )

        # Define graph network updates
        @jraph.concatenated_args
        def update_node_fn(features):
            return hk.nets.MLP([128, 128], name='node')(features)

        @jraph.concatenated_args
        def update_edge_fn(features):
            return hk.nets.MLP([128, 128], name='edge')(features)

        @jraph.concatenated_args
        def update_global_fn(features):
            return hk.nets.MLP([128, 128], name='global')(features)

        for _ in range(self.num_layers):
            # Apply the updates from the graph network
            graph_net = jraph.GraphNetwork(
                update_edge_fn=update_edge_fn,
                update_node_fn=update_node_fn,
                update_global_fn=update_global_fn,
            )
            updates = graph_net(features)

            # Apply layer normalization
            features = features._replace(
                nodes=hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True,
                )(features.nodes + updates.nodes),
                edges=hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True,
                )(features.edges + updates.edges),
                globals=hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True,
                )(features.globals + updates.globals),
            )

        # Apply self-attention layer
        node_features = features.nodes[:batch_size * num_variables]
        node_features = node_features.reshape(batch_size, num_variables, -1)
        node_features = hk.Linear(128 * 3, name='projection')(node_features)
        queries, keys, values = jnp.split(node_features, 3, axis=2)

        node_features = hk.MultiHeadAttention(
            num_heads=4,
            key_size=32,
            w_init_scale=2.,
        )(queries, keys, values)

        return Features(
            nodes=hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True
            )(node_features),
            globals=hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True
            )(features.globals[:batch_size])
        )
