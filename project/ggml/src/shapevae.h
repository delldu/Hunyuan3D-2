#ifndef __SHAPEVAE__H__
#define __SHAPEVAE__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"
extern ggml_tensor_t* scaled_dot_product_attention(struct ggml_context* ctx, ggml_tensor_t* query, ggml_tensor_t *key, ggml_tensor_t *value);

// bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
//     return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
// }


// GGML_CALL void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
//     ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

//     GGML_ASSERT(buf != NULL && "tensor buffer not set");
//     GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
//     GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

//     if (!size) {
//         return;
//     }

//     buf->iface.set_tensor(buf, tensor, data, offset, size);
// }


// def dense_grid(res, box_m=1.01):
//     # assert res == 384
//     x = torch.linspace(-box_m, box_m, res + 1)
//     y = torch.linspace(-box_m, box_m, res + 1)
//     z = torch.linspace(-box_m, box_m, res + 1)
//     xs, ys, zs = torch.meshgrid(x, y, z, indexing = "ij")
//     xyz_grid = torch.stack((xs, ys, zs), dim = -1)

//     return xyz_grid # size() -- [385, 385, 385, 3]

bool ggml_tensor_on_host(ggml_tensor_t *x) {
    ggml_backend_buffer_t buffer = x->view_src ? x->view_src->buffer : x->buffer;
    GGML_ASSERT(buffer != NULL && "tensor buffer not set");
    return ggml_backend_buffer_is_host(buffer);
}

std::vector<float> dense_grid_data(int n, int res, float box_m) {
    // res == 385 ...
    std::vector<float> data;
    float step = 2.0f * box_m / (res - 1);

    int start = n * res;
    for (int index = 0; index < res * res; index++) {
        int start_index = start + index;
        int k = start_index % res; start_index /= res;
        int j = start_index % res; start_index /= res;
        int i = start_index % res; // start_index /= res;

        data.push_back(-box_m + i * step);
        data.push_back(-box_m + j * step);
        data.push_back(-box_m + k * step);
    }

    return data;
}

bool dense_grid_fill(ggml_tensor_t *x, std::vector<float> &data) {
    if (x == nullptr || ggml_nbytes(x) != sizeof(float) * data.size())
        return false;

    if (ggml_tensor_on_host(x)) {
        memcpy(x->data, data.data(), ggml_nbytes(x));
    } else {
        ggml_backend_tensor_set(x, data.data(), 0, ggml_nbytes(x));
    }
    return true;
}


struct QKVMultiheadCrossAttention {
    int heads = 16;
    int width = 1024;

    // network params
    struct LayerNorm q_norm;
    struct LayerNorm k_norm;

    void create_weight_tensors(struct ggml_context* ctx) {
        q_norm.normalized_shape = width / heads;
        q_norm.eps = 1e-6;
        q_norm.elementwise_affine = true;
        q_norm.create_weight_tensors(ctx);

        k_norm.normalized_shape = width / heads;
        k_norm.eps = 1e-6;
        k_norm.elementwise_affine = true;
        k_norm.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "q_norm.");
        q_norm.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "k_norm.");
        k_norm.setup_weight_names(s);
    }

    // def forward(self, q, kv):
    //     # tensor [q] size: [1, 8000, 1024], min: -8.914062, max: 8.632812, mean: -0.044021
    //     # tensor [kv] size: [1, 512, 2048], min: -8.289062, max: 7.117188, mean: -0.008912
    //     _, n_q, _ = q.shape
    //     bs, n_kv, width = kv.shape
    //     attn_ch = width // self.heads // 2 # 64

    //     q = q.view(bs, n_q, self.heads, -1) # [1, 8000, 1024] --> [1, 8000, 16, 64]
    //     kv = kv.view(bs, n_kv, self.heads, -1) # [1, 512, 2048] --> [1, 512, 16, 128]
    //     k, v = torch.split(kv, attn_ch, dim=-1)
    //     # (Pdb) pp k.size() -- [1, 512, 16, 64]
    //     # (Pdb) pp v.size() -- [1, 512, 16, 64]

    //     q = self.q_norm(q)
    //     k = self.k_norm(k)

    //     # q, k, v is tuple: len = 3
    //     #     tensor [item] size: [1, 8000, 16, 64], min: -8.539062, max: 7.707031, mean: 0.039068
    //     #     tensor [item] size: [1, 512, 16, 64], min: -9.210938, max: 9.054688, mean: -0.000345
    //     #     tensor [item] size: [1, 512, 16, 64], min: -7.964844, max: 7.097656, mean: 0.005364
    //     q, k, v = map(lambda t: rearrange(t, "b n h d -> b h n d", h=self.heads), (q, k, v))
    //     # q, k, v is tuple: len = 3
    //     #     tensor [item] size: [1, 16, 8000, 64], min: -8.539062, max: 7.707031, mean: 0.039068
    //     #     tensor [item] size: [1, 16, 512, 64], min: -9.210938, max: 9.054688, mean: -0.000345
    //     #     tensor [item] size: [1, 16, 512, 64], min: -7.964844, max: 7.097656, mean: 0.005364
    //     out = F.scaled_dot_product_attention(q, k, v)
    //     out = out.transpose(1, 2).reshape(bs, n_q, -1)
    //     # tensor [out] size: [1, 8000, 1024], min: -7.214844, max: 5.949219, mean: 0.02275

    //     return out


    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* q, ggml_tensor_t* kv) {
        int n_q = (int)q->ne[1];
        int bs = (int)kv->ne[2];
        int n_kv = (int)kv->ne[1];
        int width = (int)kv->ne[0];

    //     q = q.view(bs, n_q, self.heads, -1) # [1, 8000, 1024] --> [1, 8000, 16, 64]
    //     kv = kv.view(bs, n_kv, self.heads, -1) # [1, 512, 2048] --> [1, 512, 16, 128]
        q = ggml_reshape_4d(ctx, q, -1, heads, n_q, bs);
        kv = ggml_reshape_4d(ctx, kv, -1, heads, n_kv, bs);
        int S = (int)kv->ne[0]; // 64
        ggml_tensor_t *k = ggml_nn_slice(ctx, kv, 0/*dim*/, 0*S/*start*/, 1*S/*stop*/, 1);
        ggml_tensor_t *v = ggml_nn_slice(ctx, kv, 0/*dim*/, 1*S/*start*/, 2*S/*stop*/, 1);
        q = q_norm.forward(ctx, q);
        k = q_norm.forward(ctx, k);
        // ----------------------------------------------------------------------------------------
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_permute(ctx, k, 0, 2, 1, 3);
        v = ggml_permute(ctx, v, 0, 2, 1, 3);

        ggml_tensor_dump("===> q", q);
        ggml_tensor_dump("===> k", k);
        ggml_tensor_dump("===> v", v);

        ggml_tensor_t *out = scaled_dot_product_attention(ctx, q, k, v);
        out = ggml_reshape_4d(ctx, out, 0, 2, 1, 3);
        out = ggml_reshape_3d(ctx, out, -1, n_q, bs);

        ggml_tensor_dump("===> out", out);

        return out;
    }
};

struct MultiheadCrossAttention {
    const int width = 1024;
    const int heads = 16;
    const int data_width = 1024;

    // network params
    struct Linear c_q;
    struct Linear c_kv;
    struct Linear c_proj;
    struct QKVMultiheadCrossAttention attention;

    void create_weight_tensors(struct ggml_context* ctx) {
        c_q.in_features = width;
        c_q.out_features = width;
        c_q.has_bias = false;
        c_q.create_weight_tensors(ctx);

        c_kv.in_features = data_width;
        c_kv.out_features = width * 2;
        c_kv.has_bias = false;
        c_kv.create_weight_tensors(ctx);

        c_kv.in_features = width;
        c_kv.out_features = width;
        c_kv.has_bias = true;
        c_proj.create_weight_tensors(ctx);

        attention.heads = heads;
        attention.width = width;
        attention.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "c_q.");
        c_q.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "c_kv.");
        c_kv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "c_proj.");
        c_proj.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "attention.");
        attention.setup_weight_names(s);
    }

    // def forward(self, x, data):
    //     x = self.c_q(x)
    //     data = self.c_kv(data)
    //     x = self.attention(x, data)
    //     x = self.c_proj(x)
    //     return x

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, ggml_tensor_t* data) {
        x = c_q.forward(ctx, x);
        data = c_kv.forward(ctx, data);
        x = attention.forward(ctx, x, data);
        x = c_proj.forward(ctx, x);

    	return x;
    }
};


struct MLP {
    int width = 1024;
    const int expand_ratio = 4;

    // network params
    struct Linear c_fc;
    struct Linear c_proj;

    void create_weight_tensors(struct ggml_context* ctx) {
        c_fc.in_features = width;
        c_fc.out_features = width * expand_ratio;
        c_fc.has_bias = true; // Fixed default
        c_fc.create_weight_tensors(ctx);

        c_proj.in_features = width * expand_ratio;
        c_proj.out_features = width;
        c_proj.has_bias = true; // Fixed default
        c_proj.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "c_fc.");
        c_fc.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "c_proj.");
        c_proj.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = c_fc.forward(ctx, x);
        x = ggml_gelu_quick(ctx, x);
        x = c_proj.forward(ctx, x);

        return x;
    }
};

/*
 ResidualCrossAttentionBlock(
  (attn): MultiheadCrossAttention(
    (c_q): Linear(in_features=1024, out_features=1024, bias=False)
    (c_kv): Linear(in_features=1024, out_features=2048, bias=False)
    (c_proj): Linear(in_features=1024, out_features=1024, bias=True)
    (attention): QKVMultiheadCrossAttention(
      (q_norm): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
      (k_norm): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
    )
  )
  (ln_1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
  (ln_2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
  (ln_3): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
  (mlp): MLP(
    (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
    (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
    (gelu): GELU(approximate='none')
  )
) */
struct ResidualCrossAttentionBlock {
    const int width = 1024;
    const int heads = 16;
    const int data_width = 1024;
    const int mlp_expand_ratio = 4;

    // network params
    struct MultiheadCrossAttention attn;
    struct LayerNorm ln_1;
    struct LayerNorm ln_2;
    struct LayerNorm ln_3;

    struct MLP mlp;

    void create_weight_tensors(struct ggml_context* ctx) {
        // attn.width = 1024; // width
        // attn.heads = 16; // heads
        // attn.data_width = 1024; // data_width
        attn.create_weight_tensors(ctx);

        ln_1.normalized_shape = width;
        ln_1.eps = 1e-6; // Fixed default values
        ln_1.elementwise_affine = true;
        ln_1.create_weight_tensors(ctx);

        ln_2.normalized_shape = data_width;
        ln_2.eps = 1e-6; // Fixed default values
        ln_2.elementwise_affine = true;
        ln_2.create_weight_tensors(ctx);

        ln_3.normalized_shape = width;
        ln_3.eps = 1e-6; // Fixed default values
        ln_3.elementwise_affine = true;
        ln_3.create_weight_tensors(ctx);

        mlp.width = width;
        mlp.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "attn.");
        attn.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "ln_1.");
        ln_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "ln_2.");
        ln_2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "ln_3");
        ln_3.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.");
        mlp.setup_weight_names(s);
    }


    // def forward(self, x, data):
    //     x = x + self.attn(self.ln_1(x), self.ln_2(data))
    //     x = x + self.mlp(self.ln_3(x))
    //     return x

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, ggml_tensor_t* data) {
        ggml_tensor_t *x1 = ln_1.forward(ctx, x);
        ggml_tensor_t *x2 = ln_2.forward(ctx, data);
        x1 = attn.forward(ctx, x1, x2);
        x = ggml_add(ctx, x, x1);
        ggml_tensor_t *x3 = ln_3.forward(ctx, x);
        x3 = mlp.forward(ctx, x3);
        x = ggml_add(ctx, x, x3);
    	return x;
    }
};

struct FourierEmbedder {
    const int num_freqs = 8;
    const int input_dim = 3;
    const int out_dim = 51;

    ggml_tensor_t *frequencies;

    void create_weight_tensors(struct ggml_context* ctx) {
        frequencies = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_freqs);
    }

    void setup_weight_names(const char *prefix) {
        ggml_format_name(frequencies, "%s%s", prefix, "frequencies");
    }

    // def forward(self, x):
    //     # tensor [x] size: [1, 8000, 3], min: -1.01, max: 1.01, mean: -0.65878
    //     # self.frequencies -- tensor([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128.], device='cuda:0')
    //     # (x[..., None].contiguous() * self.frequencies).size() -- [1, 8000, 3, 8]
    //     # x.shape[:-1] -- [1, 8000]
    //     embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
    //     # tensor [embed] size: [1, 8000, 24], min: -129.279999, max: 129.279999, mean: -20.998594
    //     return torch.cat((x, embed.sin(), embed.cos()), dim=-1) # [1, 8000, 51]
    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        int C0 = (int)x->ne[0]; // 3
        int C1 = (int)x->ne[1]; // 8000
        int C2 = (int)x->ne[2]; // 1
        int C3 = 1;
        ggml_tensor_t* embed = ggml_reshape_4d(ctx, x, C0, C1, C2, C3);
        ggml_tensor_dump("embed1", embed);

        embed = ggml_nn_mul_mat(ctx, embed, frequencies);
        embed = ggml_reshape_3d(ctx, embed, 24, C1, 1);
        ggml_tensor_dump("embed2", embed);

        x = ggml_cat(ctx, 3, x, ggml_sin(ctx, embed), ggml_cos(ctx, embed), 0/*dim*/);
        ggml_tensor_dump("embed3", x);

    	return x;
    }
};

struct CrossAttentionDecoder {
    const int out_channels = 1;
    const int width = 1024;
    const int heads = 16;
    const int mlp_expand_ratio = 4;
    
    // network params
    struct FourierEmbedder fourier_embedder;
    struct Linear query_proj;
    struct ResidualCrossAttentionBlock cross_attn_decoder;
    struct LayerNorm ln_post;
    struct Linear output_proj;

    void create_weight_tensors(struct ggml_context* ctx) {
        fourier_embedder.create_weight_tensors(ctx);

        query_proj.in_features = 51; // fourier_embedder.out_dim
        query_proj.out_features = width;
        query_proj.has_bias = true; // Fixed default
        query_proj.create_weight_tensors(ctx);

        // cross_attn_decoder.width = 1024; // width
        // cross_attn_decoder.int heads = 16; // heads
        // cross_attn_decoder.data_width = 1024;
        // cross_attn_decoder.mlp_expand_ratio = 4; // mlp_expand_ratio
        cross_attn_decoder.create_weight_tensors(ctx);

        ln_post.normalized_shape = width;
        ln_post.eps = 1e-5; // Fixed default values
        ln_post.elementwise_affine = true;
        ln_post.create_weight_tensors(ctx);

        output_proj.in_features = width;
        output_proj.out_features = out_channels;
        output_proj.has_bias = true; // Fixed default
        output_proj.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "fourier_embedder.");
        fourier_embedder.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "query_proj.");
        query_proj.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "cross_attn_decoder.");
        cross_attn_decoder.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "ln_post.");
        ln_post.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "output_proj.");
        output_proj.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* queries, ggml_tensor_t *latents) {
        ggml_tensor_t *query_embeddings = query_proj.forward(ctx, fourier_embedder.forward(ctx, queries));
        ggml_tensor_t *x = cross_attn_decoder.forward(ctx, query_embeddings, latents);
        x = ln_post.forward(ctx, x);
        return output_proj.forward(ctx, x);
    }
};

struct QKVMultiheadAttention {
    const int width = 1024;
    const int heads = 16;

    // network params
    struct LayerNorm q_norm;
    struct LayerNorm k_norm;

    void create_weight_tensors(struct ggml_context* ctx) {
        q_norm.normalized_shape = width/heads;
        q_norm.eps = 1e-6; // Fixed default values
        q_norm.elementwise_affine = true;
        q_norm.create_weight_tensors(ctx);

        k_norm.normalized_shape = width/heads;
        k_norm.eps = 1e-6; // Fixed default values
        k_norm.elementwise_affine = true;
        k_norm.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "q_norm.");
        q_norm.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "k_norm.");
        k_norm.setup_weight_names(s);        
    }

    // def forward(self, qkv):
    //     # tensor [qkv] size: [1, 512, 3072], min: -9.280807, max: 11.263318, mean: -0.00128

    //     bs, n_q, width = qkv.shape
    //     attn_ch = width // self.heads // 3
    //     qkv = qkv.view(bs, n_q, self.heads, -1)
    //     q, k, v = torch.split(qkv, attn_ch, dim=-1)

    //     q = self.q_norm(q)
    //     k = self.k_norm(k)
    //     # q, k, v is tuple: len = 3
    //     #     tensor [item] size: [1, 512, 16, 64], min: -6.835938, max: 6.566406, mean: -0.002259
    //     #     tensor [item] size: [1, 512, 16, 64], min: -6.929688, max: 7.53125, mean: 0.000862
    //     #     tensor [item] size: [1, 512, 16, 64], min: -3.498047, max: 3.671875, mean: -0.000247
    //     q, k, v = map(lambda t: rearrange(t, "b n h d -> b h n d", h=self.heads), (q, k, v))
    //     # q, k, v is tuple: len = 3
    //     #     tensor [item] size: [1, 16, 512, 64], min: -6.835938, max: 6.566406, mean: -0.002259
    //     #     tensor [item] size: [1, 16, 512, 64], min: -6.929688, max: 7.53125, mean: 0.000862
    //     #     tensor [item] size: [1, 16, 512, 64], min: -3.498047, max: 3.671875, mean: -0.000247

    //     # [1, 16, 512, 64] ==> [1, 512, 16, 64] ==> [1, 512, 1024]
    //     out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_q, -1)
    //     return out

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* qkv) {
        int bs = (int)qkv->ne[2];
        int n_q = (int)qkv->ne[1];
        int width = (int)qkv->ne[0];

        qkv = ggml_reshape_4d(ctx, qkv, -1, heads, n_q, bs);
        int S = (int)qkv->ne[0]/3;
        ggml_tensor_t* q = ggml_nn_slice(ctx, qkv, 0/*dim*/, 0*S/*start*/, 1*S/*stop*/, 1/*step*/);
        ggml_tensor_t* k = ggml_nn_slice(ctx, qkv, 0/*dim*/, 1*S/*start*/, 2*S/*stop*/, 1/*step*/);
        ggml_tensor_t* v = ggml_nn_slice(ctx, qkv, 0/*dim*/, 2*S/*start*/, 3*S/*stop*/, 1/*step*/);
        // --------------------------------------------------------------------------
        q = q_norm.forward(ctx, q);
        k = k_norm.forward(ctx, k);
        q = ggml_reshape_4d(ctx, q, 0, 2, 1, 3);
        k = ggml_reshape_4d(ctx, k, 0, 2, 1, 3);
        v = ggml_reshape_4d(ctx, v, 0, 2, 1, 3);

        ggml_tensor_t *out = scaled_dot_product_attention(ctx, q, k, v);
        out = ggml_reshape_4d(ctx, out, 0, 2, 1, 3);
        out = ggml_reshape_3d(ctx, out, -1, n_q, bs);

    	return out;
    }
};


struct MultiheadAttention {
    const int width = 1024;
    const int heads = 16;

    // network params
    struct Linear c_qkv;
    struct Linear c_proj;
    struct QKVMultiheadAttention attention;

    void create_weight_tensors(struct ggml_context* ctx) {
        c_qkv.in_features = width;
        c_qkv.out_features = width * 3;
        c_qkv.has_bias = false; // Fixed default
        c_qkv.create_weight_tensors(ctx);

        c_proj.in_features = width;
        c_proj.out_features = width;
        c_proj.has_bias = true; // Fixed default
        c_proj.create_weight_tensors(ctx);

        // attention.heads = 16;
        // attention.width = 1024;
        attention.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "c_qkv.");
        c_qkv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "c_proj.");
        c_proj.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "attention.");
        attention.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = c_qkv.forward(ctx, x);
        x = attention.forward(ctx, x);
        x = c_proj.forward(ctx, x);
    	return x;
    }
};

/*
 ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (c_qkv): Linear(in_features=1024, out_features=3072, bias=False)
    (c_proj): Linear(in_features=1024, out_features=1024, bias=True)
    (attention): QKVMultiheadAttention(
      (q_norm): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
      (k_norm): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
    )
  )
  (ln_1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
  (mlp): MLP(
    (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
    (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
    (gelu): GELU(approximate='none')
  )
  (ln_2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
) */


struct ResidualAttentionBlock {
    const int width = 1024;
    const int heads = 16;
    
    // network params
    struct MultiheadAttention attn;
    struct LayerNorm ln_1;
    struct MLP mlp;
    struct LayerNorm ln_2;

    void create_weight_tensors(struct ggml_context* ctx) {
        // attn.width = 1024; // width
        // attn.heads = 16; // heads
        attn.create_weight_tensors(ctx);

        ln_1.normalized_shape = width;
        ln_1.eps = 1e-6; // Fixed default values
        ln_1.elementwise_affine = true;
        ln_1.create_weight_tensors(ctx);

        mlp.width = width;
        mlp.create_weight_tensors(ctx);

        ln_2.normalized_shape = width;
        ln_2.eps = 1e-6; // Fixed default values
        ln_2.elementwise_affine = true;
        ln_2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "attn.");
        attn.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "ln_1.");
        ln_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.");
        mlp.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "ln_2.");
        ln_2.setup_weight_names(s);
    }

    //     def forward(self, x):
    //         x = x + self.attn(self.ln_1(x))
    //         x = x + self.mlp(self.ln_2(x))
    //         return x

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!
        ggml_tensor_t *x1 = ln_1.forward(ctx, x);
        x1 = attn.forward(ctx, x1);
        x = ggml_add(ctx, x, x1);

        ggml_tensor_t *x2 = ln_2.forward(ctx, x);
        x2 = mlp.forward(ctx, x2);

        x = ggml_add(ctx, x, x2);

    	return x;
    }
};

/*
 Transformer(
  (resblocks): ModuleList(
    (0-15): 16 x ResidualAttentionBlock(
      (attn): MultiheadAttention(
        (c_qkv): Linear(in_features=1024, out_features=3072, bias=False)
        (c_proj): Linear(in_features=1024, out_features=1024, bias=True)
        (attention): QKVMultiheadAttention(
          (q_norm): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
          (k_norm): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        )
      )
      (ln_1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
      (mlp): MLP(
        (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
        (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
        (gelu): GELU(approximate='none')
      )
      (ln_2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
    )
  )
) */

struct Transformer {
    const int width = 1024;
    const int heads = 16;
    // const int layers = 16;

    // network params
    struct ResidualAttentionBlock resblocks[16];

    void create_weight_tensors(struct ggml_context* ctx) {
        for (int i = 0; i < 16; i++) {
            // resblocks[i].width = 1024; // width
            // resblocks[i].heads = 16; // heads
            resblocks[i].create_weight_tensors(ctx);
        }
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        for (int i = 0; i < 16; i++) {
            snprintf(s, sizeof(s), "%sresblocks.%d.", prefix, i);
            resblocks[i].setup_weight_names(s);
        }
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        for (int i = 0; i < 16; i++) {
            x = resblocks[i].forward(ctx, x);
        }
    	return x;
    }
};


struct ShapeVAE {
    const int num_latents = 512;
    const int embed_dim = 64;
    const int width = 1024;
    const int heads = 16;
    // const int num_decoder_layers = 16;

    // network hparams
    float scale_factor = 1.0188137142395404;

    // network params
    struct Linear post_kl;
    struct Transformer transformer;
    struct CrossAttentionDecoder geo_decoder;

    void create_weight_tensors(struct ggml_context* ctx) {
        post_kl.in_features = embed_dim;
        post_kl.out_features = width;
        post_kl.has_bias = true; // Fixed default
        post_kl.create_weight_tensors(ctx);

        // transformer.width = 1024; // width
        // transformer.heads = 16; // heads
        transformer.create_weight_tensors(ctx);

        // geo_decoder.out_channels = 1; // out_channels
        // geo_decoder.width = 1024; // width
        // geo_decoder.heads = 16; // heads
        // geo_decoder.mlp_expand_ratio = 4; // mlp_expand_ratio
        // geo_decoder.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "post_kl.");
        post_kl.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "transformer.");
        transformer.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "geo_decoder.");
        geo_decoder.setup_weight_names(s);
    }

    // def forward(self, latents):
    //     # 1. latents decode
    //     latents = latents/self.scale_factor 
    //     # tensor [latents] size: [1, 512, 64], min: -4.003906, max: 3.90625, mean: 0.018309
    //     latents = self.post_kl(latents)
    //     latents = self.transformer(latents)
    //     # tensor [latents] size: [1, 512, 1024], min: -374.5, max: 37.09375, mean: 0.019848

    //     # 2. latents to 3d volume
    //     grid_res = 384
    //     num_chunks = 8000
    //     batch_size = latents.shape[0] # 1
    //     xyz_samples = dense_grid(384)
    //     xyz_samples = xyz_samples.view(-1, 3).to(latents.device)
    //     # tensor [xyz_samples] size: [57066625, 3], min: -1.009766, max: 1.009766, mean: 0.0

    //     # running on cuda device
    //     batch_logits = []
    //     for start in tqdm(range(0, xyz_samples.shape[0], num_chunks), desc=f"Volume Decoding"):
    //         chunk_queries = xyz_samples[start: start + num_chunks, :]
    //         chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
    //         logits = self.geo_decoder(queries=chunk_queries, latents=latents)
    //         batch_logits.append(logits)
    //     # len(batch_logits) -- 7134
    //     # batch_logits[0].size() -- [1, 8000, 1]
    //     grid_logits = torch.cat(batch_logits, dim=1) # torch.cat(batch_logits, dim=1).size() -- [1, 57066625, 1]
    //     # grid_size --[385, 385, 385]
    //     grid_logits = grid_logits.view((batch_size, grid_res + 1, grid_res + 1, grid_res + 1)).float()
    //     # 385*385*385 === 57066625
    //     # tensor [grid_logits] size: [1, 385, 385, 385], min: -1.082031, max: 1.067383, mean: -0.787309
    //     return grid_logits


    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* latents) {
        // 1. latents decode
        latents = ggml_scale(ctx, latents, scale_factor);
        latents = post_kl.forward(ctx, latents);
        latents = transformer.forward(ctx, latents);

        // 2. latents to 3d volume
        int grid_res = 385;
        int num_chunks = 8 * 1024;
        int batch_size = latents->ne[2]; // === 1
        GGML_ASSERT(batch_size == 1);


        // 3. running geo_decoder
        ggml_tensor_t *chunk_queries = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, grid_res*grid_res, 1);
        ggml_tensor_t *logits;
        ggml_tensor_t *grid_logits = nullptr;

        for (int n = 0; n < grid_res; n++) {
            std::vector<float> data = dense_grid_data(n, grid_res, 1.01);
            dense_grid_fill(chunk_queries, data);

            ggml_tensor_dump("chunk_queries", chunk_queries);
            logits = geo_decoder.forward(ctx, chunk_queries, latents);

            if (grid_logits == nullptr) {
                grid_logits = ggml_dup(ctx, logits);
            } else {
                grid_logits = ggml_concat(ctx, grid_logits, logits, 1/*dim*/);
            }
        }

        grid_logits = ggml_reshape_4d(ctx, grid_logits, grid_res, grid_res, grid_res, 3);
        ggml_tensor_dump("grid_logits", grid_logits);

    	return grid_logits;
    }
};

#endif // __SHAPEVAE__H__
