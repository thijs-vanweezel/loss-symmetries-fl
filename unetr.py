# Mostly based on https://docs.jaxstack.ai/en/latest/JAX_examples_image_segmentation.html
from jax import numpy as jnp
from flax import nnx
import jax
from typing import Callable
from models import AsymConv, AsymLinear

class AsymConvTranspose(AsymConv):
    def __call__(self, inputs:jax.Array) -> jax.Array:
        bias = self.bias
        kernel = self.kernel
        # Apply SyRe (before wasym to avoid biasing the masked weights)
        if self.ssigma>0.:
            if self.use_bias: bias = bias + self.randb * self.ssigma
            kernel = kernel + self.randk * self.ssigma
        # Order bias to counter permutation symmetry
        if self.orderbias: bias = jnp.concatenate([
            bias[0:1],
            jnp.cumsum(jnp.exp(bias[1:])) + bias[0:1]
        ])
        # Apply w-asymmetry
        if self.wasym:
            kernel = kernel * self.wmask + (1-self.wmask) * self.randk * self.kappa
        # Normalize kernel to unit norm per output neuron
        if self.normweights:
            norm = jnp.linalg.norm(kernel.reshape(-1, self.out_features), axis=0, keepdims=True)
            kernel /= norm
            if self.use_bias: bias /= norm.squeeze()
        # Implementation directly copied from nnx.Conv
        inputs, kernel, bias = self.promote_dtype(
            (inputs, kernel, bias), dtype=self.dtype
        )
        y = jax.lax.conv_transpose(
            inputs,
            kernel,
            self.maybe_broadcast(self.strides),
            self.padding,
            rhs_dilation=self.maybe_broadcast(self.kernel_dilation),
            transpose_kernel=False,
            precision=self.precision,
        )
        if self.use_bias:
            y += bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
        return y

class PatchEmbeddingBlock(nnx.Module):
    """
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """
    def __init__(
        self,
        in_channels: int,  # dimension of input channels.
        img_size: int,  # dimension of input image.
        patch_size: int,  # dimension of patch size.
        hidden_size: int,  # dimension of hidden layer.
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
        **asymkwargs
    ):
        n_patches = (img_size // patch_size) ** 2
        self.patch_embeddings = AsymConv(
            in_channels,
            hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
            param_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            **asymkwargs
        )

        initializer = jax.nn.initializers.truncated_normal(stddev=0.02)
        self.position_embeddings = nnx.Param(
            initializer(rngs.params(), (1, n_patches, hidden_size), jnp.bfloat16)
        )
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.patch_embeddings(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class MLPBlock(nnx.Sequential):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """
    def __init__(
        self,
        hidden_size: int,  # dimension of hidden layer.
        mlp_dim: int,      # dimension of feedforward layer
        dropout_rate: float = 0.0,
        activation_layer: Callable = nnx.gelu,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
        **asymkwargs
    ):
        layers = [
            AsymLinear(hidden_size, mlp_dim, rngs=rngs, param_dtype=jnp.bfloat16, dtype=jnp.bfloat16, **asymkwargs),
            activation_layer,
            nnx.Dropout(dropout_rate, rngs=rngs),
            AsymLinear(mlp_dim, hidden_size, rngs=rngs, param_dtype=jnp.bfloat16, dtype=jnp.bfloat16, **asymkwargs),
            nnx.Dropout(dropout_rate, rngs=rngs),
        ]
        super().__init__(*layers)

class ViTEncoderBlock(nnx.Module):
    """
    A transformer encoder block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """
    def __init__(
        self,
        hidden_size: int,  # dimension of hidden layer.
        mlp_dim: int,      # dimension of feedforward layer.
        num_heads: int,    # number of attention heads
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
        **asymkwargs
    ) -> None:
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate, rngs=rngs, **asymkwargs)
        self.norm1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            dropout_rate=dropout_rate,
            broadcast_dropout=False,
            decode=False,
            rngs=rngs,
            param_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16
        )
        self.norm2 = nnx.LayerNorm(hidden_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nnx.Module):
    """
    Vision Transformer (ViT) Feature Extractor, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """
    def __init__(
        self,
        in_channels: int,  # dimension of input channels
        img_size: int,  # dimension of input image
        patch_size: int,  # dimension of patch size
        hidden_size: int = 768,  # dimension of hidden layer
        mlp_dim: int = 3072,  # dimension of feedforward layer
        num_layers: int = 12,  # number of transformer blocks
        num_heads: int = 12,   # number of attention heads
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
        **asymkwargs
    ):
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            rngs=rngs,
            **asymkwargs
        )
        self.blocks = [
            ViTEncoderBlock(hidden_size, mlp_dim, num_heads, dropout_rate, rngs=rngs, **asymkwargs)
            for i in range(num_layers)
        ]
        self.norm = nnx.LayerNorm(hidden_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.patch_embedding(x)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        return x, hidden_states_out

class Conv2dNormActivation(nnx.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        norm_layer: Callable[..., nnx.Module] = nnx.BatchNorm,
        activation_layer: Callable = nnx.relu,
        dilation: int = 1,
        bias: bool | None = None,
        rngs: nnx.Rngs = nnx.Rngs(0),
        **asymkwargs
    ):
        self.out_channels = out_channels

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        # sequence integer pairs that give the padding to apply before
        # and after each spatial dimension
        padding = ((padding, padding), (padding, padding))

        layers = [
            AsymConv(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                strides=(stride, stride),
                padding=padding,
                kernel_dilation=(dilation, dilation),
                feature_group_count=groups,
                use_bias=bias,
                rngs=rngs,
                param_dtype=jnp.bfloat16,
                dtype=jnp.bfloat16,
                **asymkwargs
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels, rngs=rngs))

        if activation_layer is not None:
            layers.append(activation_layer)

        super().__init__(*layers)

class InstanceNorm(nnx.GroupNorm):
    def __init__(self, num_features, **kwargs):
        num_groups, group_size = num_features, None
        super().__init__(
            num_features,
            num_groups=num_groups,
            group_size=group_size,
            **kwargs,
        )


class UnetResBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_layer: Callable[..., nnx.Module] = InstanceNorm,
        activation_layer: Callable = nnx.leaky_relu,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
        **asymkwargs
    ):
        self.conv_norm_act1 = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            rngs=rngs,
            **asymkwargs
        )
        self.conv_norm2 = Conv2dNormActivation(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=None,
            rngs=rngs,
            **asymkwargs
        )

        self.downsample = (in_channels != out_channels) or (stride != 1)
        if self.downsample:
            self.conv_norm3 = Conv2dNormActivation(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=None,
                rngs=rngs,
                **asymkwargs
            )
        self.act = activation_layer

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        out = self.conv_norm_act1(x)
        out = self.conv_norm2(out)
        if self.downsample:
            residual = self.conv_norm3(residual)
        out += residual
        out = self.act(out)
        return out

class UnetrBasicBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_layer: Callable[..., nnx.Module] = InstanceNorm,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
        **asymkwargs
    ):
        self.layer = UnetResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_layer=norm_layer,
            rngs=rngs,
            **asymkwargs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.layer(x)

class UnetrPrUpBlock(nnx.Module):
    """
    A projection upsampling module for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,  # number of input channels.
        out_channels: int, # number of output channels.
        num_layer: int,    # number of upsampling blocks.
        kernel_size: int,
        stride: int,
        upsample_kernel_size: int = 2,  # convolution kernel size for transposed convolution layers.
        norm_layer: Callable[..., nnx.Module] = InstanceNorm,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
        **asymkwargs
    ):
        upsample_stride = upsample_kernel_size
        self.transp_conv_init = AsymConvTranspose(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(upsample_kernel_size, upsample_kernel_size),
            strides=(upsample_stride, upsample_stride),
            padding="VALID",
            rngs=rngs,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            **asymkwargs
        )
        self.blocks = [
            nnx.Sequential(
                AsymConvTranspose(
                    in_features=out_channels,
                    out_features=out_channels,
                    kernel_size=(upsample_kernel_size, upsample_kernel_size),
                    strides=(upsample_stride, upsample_stride),
                    rngs=rngs,
                    dtype=jnp.bfloat16,
                    param_dtype=jnp.bfloat16,
                    **asymkwargs
                ),
                UnetResBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    norm_layer=norm_layer,
                    rngs=rngs,
                    **asymkwargs
                ),
            )
            for _ in range(num_layer)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class UnetrUpBlock(nnx.Module):
    """
    An upsampling module for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        upsample_kernel_size: int = 2,  # convolution kernel size for transposed convolution layers.
        norm_layer: Callable[..., nnx.Module] = InstanceNorm,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
        **asymkwargs
    ) -> None:
        upsample_stride = upsample_kernel_size
        self.transp_conv = AsymConvTranspose(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(upsample_kernel_size, upsample_kernel_size),
            strides=(upsample_stride, upsample_stride),
            padding="VALID",
            rngs=rngs,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            **asymkwargs
        )
        self.conv_block = UnetResBlock(
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_layer=norm_layer,
            rngs=rngs,
            **asymkwargs
        )

    def __call__(self, x: jax.Array, skip: jax.Array) -> jax.Array:
        out = self.transp_conv(x)
        out = jnp.concat((out, skip), axis=-1)
        out = self.conv_block(out)
        return out

class UNETR(nnx.Module):
    """UNETR model ported to NNX from MONAI implementation:
    - https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py
    """
    def __init__(
        self,
        out_channels: int,
        in_channels: int = 3,
        img_size: int = 256,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        norm_layer: Callable[..., nnx.Module] = InstanceNorm,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
        **asymkwargs
    ):
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        self.patch_size = 16
        self.feat_size = img_size // self.patch_size
        self.hidden_size = hidden_size

        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            rngs=rngs,
            **asymkwargs
        )
        self.encoder1 = UnetrBasicBlock(
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_layer=norm_layer,
            rngs=rngs,
            **asymkwargs
        )
        self.encoder2 = UnetrPrUpBlock(
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_layer=norm_layer,
            rngs=rngs,
            **asymkwargs
        )
        self.encoder3 = UnetrPrUpBlock(
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_layer=norm_layer,
            rngs=rngs,
            **asymkwargs
        )
        self.encoder4 = UnetrPrUpBlock(
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_layer=norm_layer,
            rngs=rngs,
            **asymkwargs
        )
        self.decoder5 = UnetrUpBlock(
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_layer=norm_layer,
            rngs=rngs,
            **asymkwargs
        )
        self.decoder4 = UnetrUpBlock(
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_layer=norm_layer,
            rngs=rngs,
            **asymkwargs
        )
        self.decoder3 = UnetrUpBlock(
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_layer=norm_layer,
            rngs=rngs,
            **asymkwargs
        )
        self.decoder2 = UnetrUpBlock(
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_layer=norm_layer,
            rngs=rngs,
            **asymkwargs
        )

        self.out = AsymConv(
            in_features=feature_size,
            out_features=out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            **asymkwargs
        )

        self.proj_axes = (0, 1, 2, 3)
        self.proj_view_shape = [self.feat_size, self.feat_size, self.hidden_size]

    def proj_feat(self, x: jax.Array) -> jax.Array:
        new_view = [x.shape[0]] + self.proj_view_shape
        x = x.reshape(new_view)
        x = jnp.permute_dims(x, self.proj_axes)
        return x

    def __call__(self, x_in: jax.Array, train:bool=None) -> jax.Array:
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)