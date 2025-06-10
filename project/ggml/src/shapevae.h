#ifndef __SHAPEVAE__H__
#define __SHAPEVAE__H__
// #include "ggml_engine.h"
#include "ggml_model.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"
extern ggml_tensor_t* scaled_dot_product_attention(struct ggml_context* ctx, ggml_tensor_t* query, ggml_tensor_t *key, ggml_tensor_t *value);

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

        // q = q.view(bs, n_q, self.heads, -1) # [1, 8000, 1024] --> [1, 8000, 16, 64]
        // v = kv.view(bs, n_kv, self.heads, -1) # [1, 512, 2048] --> [1, 512, 16, 128]
        q = ggml_reshape_4d(ctx, q, -1, heads, n_q, bs);
        kv = ggml_reshape_4d(ctx, kv, -1, heads, n_kv, bs);
        int S = (int)kv->ne[0]/2; // 64
        ggml_tensor_t *k = ggml_nn_slice(ctx, kv, 0/*dim*/, 0*S/*start*/, 1*S/*stop*/, 1);
        ggml_tensor_t *v = ggml_nn_slice(ctx, kv, 0/*dim*/, 1*S/*start*/, 2*S/*stop*/, 1);

        // ----------------------------------------------------------------------------------------
        q = q_norm.forward(ctx, q);
        k = q_norm.forward(ctx, k);
        // q    f32 [64, 16, 148225, 1], 
        // k    f32 [64, 16, 512, 1], 
        // v    f32 [64, 16, 512, 1],  (reshaped) (view) (cont)
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        // q    f32 [64, 148225, 16, 1],  (permuted) (cont)
        // k    f32 [64, 512, 16, 1],  (permuted) (cont)
        // v    f32 [64, 512, 16, 1],  (reshaped) (view) (cont) (permuted) (cont)
        // ----------------------------------------------------------------------------------------
        ggml_tensor_t *out = scaled_dot_product_attention(ctx, q, k, v);
        out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
        out = ggml_reshape_3d(ctx, out, -1, n_q, bs);
        //  out    f32 [1024, 148225, 1, 1],  (permuted) (cont) (reshaped)

        return out;
    }
};

struct MultiheadCrossAttention {
    const int width = 1024;
    const int heads = 16;
    const int data_width = 1024;

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

        c_proj.in_features = width;
        c_proj.out_features = width;
        c_proj.has_bias = true;
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

    struct Linear c_fc;
    struct Linear c_proj;

    void create_weight_tensors(struct ggml_context* ctx) {
        c_fc.in_features = width;
        c_fc.out_features = width * expand_ratio;
        c_fc.has_bias = true;
        c_fc.create_weight_tensors(ctx);

        c_proj.in_features = width * expand_ratio;
        c_proj.out_features = width;
        c_proj.has_bias = true;
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
        snprintf(s, sizeof(s), "%s%s", prefix, "ln_3.");
        ln_3.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.");
        mlp.setup_weight_names(s);
    }

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

    // ggml_tensor_t *frequencies;

    void create_weight_tensors(struct ggml_context* ctx) {
        GGML_UNUSED(ctx);
        // frequencies = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, num_freqs, 1);
    }

    void setup_weight_names(const char *prefix) {
        GGML_UNUSED(prefix);
        // ggml_format_name(frequencies, "%s%s", prefix, "frequencies");
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
        // x    f32 [3, 148225, 1, 1], 

        int C0 = (int)x->ne[0]; // 3
        int C1 = (int)x->ne[1]; // 8000
        int C2 = (int)x->ne[2]; // 1
        // int C3 = 1;
        ggml_tensor_t* embed = ggml_reshape_4d(ctx, x, 1, C0, C1, C2);
        embed = ggml_repeat_ext(ctx, embed, 8, 1, 1, 1);
        // embed1    f32 [8, 3, 148225, 1], 

        ggml_tensor_t *frequencies = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, num_freqs, 1);
        // frequencies    f32 [8, 1, 1, 1], 

        float data[8] = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f};
        ggml_backend_tensor_set(frequencies, data, 0, ggml_nbytes(frequencies));

        embed = ggml_mul(ctx, embed, frequencies); // Dot
        // embed f32 [8, 3, 148225, 1],

        embed = ggml_reshape_3d(ctx, embed, 24, C1, 1);
        // embed    f32 [24, 148225, 1, 1],  (reshaped)

        x = ggml_cat(ctx, 3, x, ggml_sin(ctx, embed), ggml_cos(ctx, embed), 0/*dim*/);
        // embed f32 [51, 148225, 1, 1],

    	return x;
    }
};


struct QKVMultiheadAttention {
    const int width = 1024;
    const int heads = 16;

    struct LayerNorm q_norm;
    struct LayerNorm k_norm;

    void create_weight_tensors(struct ggml_context* ctx) {
        q_norm.normalized_shape = width/heads;
        q_norm.eps = 1e-6;
        q_norm.elementwise_affine = true;
        q_norm.create_weight_tensors(ctx);

        k_norm.normalized_shape = width/heads;
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
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

        ggml_tensor_t *out = scaled_dot_product_attention(ctx, q, k, v);
        out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
        out = ggml_reshape_3d(ctx, out, -1, n_q, bs);

    	return out;
    }
};


struct MultiheadAttention {
    const int width = 1024;
    const int heads = 16;

    struct Linear c_qkv;
    struct Linear c_proj;
    struct QKVMultiheadAttention attention;

    void create_weight_tensors(struct ggml_context* ctx) {
        c_qkv.in_features = width;
        c_qkv.out_features = width * 3;
        c_qkv.has_bias = false;
        c_qkv.create_weight_tensors(ctx);

        c_proj.in_features = width;
        c_proj.out_features = width;
        c_proj.has_bias = true;
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

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
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

    struct ResidualAttentionBlock resblocks[16]; // const int layers = 16;

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

// struct ShapeVAE
struct ShapeVaeNetwork : ggml::GGMLNetwork {
    const int num_latents = 512;
    const int embed_dim = 64;
    const int width = 1024;
    const int heads = 16;
    const int grid_res = 385;
    const float scale_factor = 1.0188137142395404;

    struct Linear post_kl;
    struct Transformer transformer;

    size_t get_graph_size()
    {
        return 16*GGML_DEFAULT_GRAPH_SIZE; // 2048
    }


    void create_weight_tensors(struct ggml_context* ctx) {
        post_kl.in_features = embed_dim;
        post_kl.out_features = width;
        post_kl.has_bias = true; // Fixed default
        post_kl.create_weight_tensors(ctx);

        // transformer.width = 1024; // width
        // transformer.heads = 16; // heads
        transformer.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "post_kl.");
        post_kl.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "transformer.");
        transformer.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_ASSERT(argc == 1);
        ggml_tensor_t *latents = argv[0];

        //     # tensor [latents] size: [1, 512, 64], min: -4.003906, max: 3.90625, mean: 0.018309
        latents = ggml_scale(ctx, latents, scale_factor);
        latents = post_kl.forward(ctx, latents);
        latents = transformer.forward(ctx, latents);

        // tensor [latents] size: [1, 512, 1024], min: -374.5, max: 37.09375, mean: 0.019848
        return latents;
    }
};


struct GeoDecoder : ggml::GGMLNetwork {
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

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_ASSERT(argc == 2);
        ggml_tensor_t *queries = argv[0];
        ggml_tensor_t *latents = argv[1];

        ggml_tensor_t *x;
        x = fourier_embedder.forward(ctx, queries);
        x = query_proj.forward(ctx, x);
        x = cross_attn_decoder.forward(ctx, x, latents);
        x = ln_post.forward(ctx, x);
        x = output_proj.forward(ctx, x);

        return x;
    }
};


// struct ShapeVaeModel {
//     ShapeVaeNetwork shape_vae_net;

//     int init(int device)
//     {
//         GGMLModel model;

//         shape_vae_net.set_device(device);
//         shape_vae_net.start_engine();
//         shape_vae_net.dump();
//         check_point(model.preload("models/image3d_shape.gguf") == RET_OK);
//         shape_vae_net.load_weight(&model, "shape_vae.");
//         model.clear();

//         return RET_OK;
//     }

//     TENSOR* forward(TENSOR* latents)
//     {
//         TENSOR* argv[1];
//         argv[0] = latents;

//         // # tensor [latents] size: [1, 512, 64], min: -4.003906, max: 3.90625, mean: 0.018309
//         TENSOR* y = shape_vae_net.engine_forward(ARRAY_SIZE(argv), argv);
//         // # tensor [latents] size: [1, 512, 1024], min: -374.5, max: 37.09375, mean: 0.019848

//         return y;
//     }

//     void exit()
//     {
//         shape_vae_net.stop_engine();
//     }
// };


// struct GeoDecoderModel {
//     GeoDecoder geo_decorde_net;

//     int init(int device)
//     {
//         ggml::GGMLModel model;

//         // -----------------------------------------------------------------------------------------
//         geo_decorde_net.set_device(device);
//         geo_decorde_net.start_engine();
//         geo_decorde_net.dump();
//         check_point(model.preload("models/image3d_shape.gguf") == RET_OK);
//         // load weights ...
//         geo_decorde_net.load_weight(&model, "shape_vae.geo_decoder.");
//         model.clear();

//         return RET_OK;
//     }

//     TENSOR* forward(TENSOR *queries, TENSOR* latents)
//     {
//         TENSOR* argv[1];
//         argv[0] = queries;
//         argv[1] = latents;

//         // # tensor [queries] size: [1, 8000, 3], min: -1.009766, max: 1.009766, mean: -0.658704
//         // # tensor [latents] size: [1, 512, 1024], min: -369.75, max: 36.4375, mean: 0.016268
//         TENSOR* y = geo_decorde_net.engine_forward(ARRAY_SIZE(argv), argv);
//         // # tensor [y] size: [1, 8000, 1], min: -1.000977, max: -0.999023, mean: -0.999919

//         return y;
//     }

//     void exit()
//     {
//         geo_decorde_net.stop_engine();
//     }
// };
#endif // __SHAPEVAE__H__
