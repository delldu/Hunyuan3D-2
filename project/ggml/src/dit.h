#ifndef __HUNYUAN3DDIT__H__
#define __HUNYUAN3DDIT__H__
// #include "ggml_engine.h"
#include "ggml_model.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"

extern ggml_tensor_t* scaled_dot_product_attention(struct ggml_context* ctx, ggml_tensor_t* query, ggml_tensor_t *key, ggml_tensor_t *value);


ggml_tensor_t* attention(struct ggml_context* ctx, ggml_tensor_t* q, ggml_tensor_t* k, ggml_tensor_t* v)
{
    ggml_tensor *x = scaled_dot_product_attention(ctx, q, k, v);

    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3)); // [64, 1882, 16, 2] ==> [64, 16, 1882, 2] ==> [1024, 1882, 2]

    int C0 = (int)x->ne[0];
    int C1 = (int)x->ne[1];
    int C2 = (int)x->ne[2];
    int C3 = (int)x->ne[3];
    x = ggml_reshape_3d(ctx, x, C0 * C1, C2, C3); // [1024, 1882, 2]

    return x;
}

// def timestep_embedding(t, dim=256, max_period=1000, time_factor=1000.0):
//     assert dim == 256
//     assert max_period == 1000
//     assert time_factor == 1000

//     t = time_factor * t
//     half = dim // 2
//     freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
//     # ============================================================
//     # math.log(max_period) === 6.907755278982137
//     # math.log(max_period)/half === 0.053966838117047944
//     # ============================================================
//     # freqs.size() -- [128]
//     # freqs
//     # tensor([1.000000, 0.947464, 0.897687, 0.850526, 0.805842, 0.763506, 0.723394,
//     #         0.685390, 0.649382, 0.615265, 0.582942, 0.552316, 0.523299, 0.495807,
//     #         0.469759, 0.445079, 0.421697, 0.399542, 0.378552, 0.358664, 0.339821,
//     #         0.321968, 0.305053, 0.289026, 0.273842, 0.259455, 0.245824, 0.232910,
//     #         0.220673, 0.209080, 0.198096, 0.187688, 0.177828, 0.168485, 0.159634,
//     #         0.151247, 0.143301, 0.135773, 0.128640, 0.121881, 0.115478, 0.109411,
//     #         0.103663, 0.098217, 0.093057, 0.088168, 0.083536, 0.079148, 0.074989,
//     #         0.071050, 0.067317, 0.063780, 0.060430, 0.057255, 0.054247, 0.051397,
//     #         0.048697, 0.046138, 0.043714, 0.041418, 0.039242, 0.037180, 0.035227,
//     #         0.033376, 0.031623, 0.029961, 0.028387, 0.026896, 0.025483, 0.024144,
//     #         0.022876, 0.021674, 0.020535, 0.019456, 0.018434, 0.017466, 0.016548,
//     #         0.015679, 0.014855, 0.014075, 0.013335, 0.012635, 0.011971, 0.011342,
//     #         0.010746, 0.010182, 0.009647, 0.009140, 0.008660, 0.008205, 0.007774,
//     #         0.007365, 0.006978, 0.006612, 0.006264, 0.005935, 0.005623, 0.005328,
//     #         0.005048, 0.004783, 0.004532, 0.004294, 0.004068, 0.003854, 0.003652,
//     #         0.003460, 0.003278, 0.003106, 0.002943, 0.002788, 0.002642, 0.002503,
//     #         0.002371, 0.002247, 0.002129, 0.002017, 0.001911, 0.001811, 0.001715,
//     #         0.001625, 0.001540, 0.001459, 0.001382, 0.001310, 0.001241, 0.001176,
//     #         0.001114, 0.001055], device='cuda:0')
//     # pp t.size() -- [2], t[:, None].size() -- [2, 1], # freqs[None].size() -- [1, 128]
//     args = t[:, None].float() * freqs[None]
//     # args.size() -- [2, 128]
//     # torch.cos(args).size() -- [2, 128]
//     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
//     if dim % 2: # False
//         pdb.set_trace()
//         embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
//     if torch.is_floating_point(t):  # True
//         embedding = embedding.to(t)

//     # tensor [embedding] size: [2, 256], min: 0.0, max: 1.0, mean: 0.5
//     return embedding


ggml_tensor_t* timestep_embedding(struct ggml_context* ctx, ggml_tensor_t* t, int dim /*=256*/, float time_factor /*=1000.0*/)
{
    int half = dim/2;
    t = ggml_scale(ctx, t, time_factor);

    // max_period = 1000.0f;
    float scale = -0.053966838117047944f; // -math.log(max_period)/half

    // freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
    ggml_tensor_t* freqs = ggml_arange(ctx, 0.0f, (float)half, 1.0f);
    freqs = ggml_scale(ctx, freqs, scale);
    freqs = ggml_exp(ctx, freqs);

    t = ggml_reshape_2d(ctx, t, 1, 2);
    freqs = ggml_reshape_2d(ctx, freqs, 128, 1);
    freqs = ggml_repeat_ext(ctx, freqs, 1, 2, 1, 1);
    // t    f32 [1, 2, 1, 1],  (reshaped)
    // freqs    f32 [128, 2, 1, 1],  (reshaped)

    ggml_tensor_t *args = ggml_mul(ctx, freqs, t); // Dot
    ggml_tensor_t *out = ggml_concat(ctx, ggml_cos(ctx, args), ggml_sin(ctx, args), 0/*dim*/);

    return out; // out f32 [256, 2, 1, 1]
}


struct LastLayer {
    const int hidden_size = 1024;
    const int patch_size = 1;
    const int out_channels = 64;

    struct LayerNorm norm_final;
    struct Linear linear;
    struct Linear adaLN_modulation_1;

    void create_weight_tensors(struct ggml_context* ctx) {
        norm_final.normalized_shape = hidden_size;
        norm_final.eps = 1e-6; // Fixed default values
        norm_final.elementwise_affine = false;
        norm_final.create_weight_tensors(ctx);

        linear.in_features = hidden_size;
        linear.out_features = patch_size * patch_size * out_channels;
        linear.has_bias = true; // Fixed default
        linear.create_weight_tensors(ctx);

        adaLN_modulation_1.in_features = hidden_size;
        adaLN_modulation_1.out_features = 2 * hidden_size;
        adaLN_modulation_1.has_bias = true; // Fixed default
        adaLN_modulation_1.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        // snprintf(s, sizeof(s), "%s%s", prefix, "norm_final.");
        // norm_final.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "linear.");
        linear.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "adaLN_modulation.1.");
        adaLN_modulation_1.setup_weight_names(s);
    }


    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, ggml_tensor_t* vec) {
        // vec    f32 [1024, 2, 1, 1], 

        //  1. shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        vec = ggml_silu(ctx, vec);
        vec = adaLN_modulation_1.forward(ctx, vec); // vec    f32 [2048, 2, 1, 1],
        ggml_tensor_t *shift = ggml_nn_slice(ctx, vec, 0/*dim*/, 0 /*start*/, hidden_size, 1/*step*/);
        ggml_tensor_t *scale = ggml_nn_slice(ctx, vec, 0/*dim*/, hidden_size /*start*/, 2*hidden_size, 1/*step*/);
        // shift    f32 [1024, 2, 1, 1],  (view) (cont)
        // scale    f32 [1024, 2, 1, 1],  (view) (cont)

        // 2. x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        int C0 = (int)shift->ne[0];
        int C1 = (int)shift->ne[1];
        scale = ggml_reshape_3d(ctx, scale, C0, 1, C1);
        shift = ggml_reshape_3d(ctx, shift, C0, 1, C1);
        // scale    f32 [1024, 1, 2, 1],  (view) (cont) (reshaped)
        // shift    f32 [1024, 1, 2, 1],  (view) (cont) (reshaped)

        scale = ggml_add_constant(ctx, scale, 1.0f);
        x = norm_final.forward(ctx, x);
        // shift    f32 [1024, 1, 2, 1],  (view) (cont) (reshaped)

        // x    f32 [1024, 512, 2, 1], 
        // scale    f32 [1024, 1, 2, 1], 
        x = ggml_mul(ctx, x, scale); // Dot !!!
        x = ggml_add(ctx, x, shift);

        // 3. x = self.linear(x)
        x = linear.forward(ctx, x);
        // x    f32 [64, 512, 2, 1], 

    	return x;
    }
};

struct SingleModulation {
    const int dim = 1024;
    const int multiplier = 3;

    struct Linear lin;

    void create_weight_tensors(struct ggml_context* ctx) {
        lin.in_features = dim;
        lin.out_features = multiplier * dim;
        lin.has_bias = true; // Fixed default
        lin.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "lin.");
        lin.setup_weight_names(s);
    }

    //     def forward(self, vec):
    //         out = F.silu(vec)
    //         # tensor [out] size: [2, 1024], min: -0.118526, max: 5.067151, mean: 0.023684
    //         out = self.lin(out)
    //         # tensor [out] size: [2, 3072], min: -2.356169, max: 2.526265, mean: 0.078408

    //         out = out[:, None, :] # size() -- [2, 1, 3072]
    //         out = out.chunk(self.multiplier, dim=-1)

    //         return out[0], out[1], out[2] # shift, scale, gate

    std::vector<ggml_tensor_t*> forward(struct ggml_context* ctx, ggml_tensor_t* vec) {
        ggml_tensor_t *out = ggml_silu(ctx, vec);
        out = lin.forward(ctx, out);

        int C0 = (int)out->ne[0];
        int C1 = (int)out->ne[1];
        out = ggml_reshape_3d(ctx, out, C0, 1, C1);
        // out f32 [3072, 1, 2, 1],  (reshaped)

        return ggml_nn_chunks(ctx, out, 0/*dim*/, multiplier); // shift, scale, gate
    }
};


struct RMSNorm {
    int dim = 64;
    
    ggml_tensor_t* scale;

    void create_weight_tensors(struct ggml_context* ctx) {
        scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    }

    void setup_weight_names(const char *prefix) {
        ggml_format_name(scale, "%s%s", prefix, "scale");
    }

    // def forward(self, x):
    //     # x.size() -- [2, 16, 512, 64]
    //     # torch.mean(x ** 2, dim=-1, keepdim=True).size() -- 2, 16, 512, 1]

    //     rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
    //     return (x * rrms).to(dtype=x_dtype) * self.scale


    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // ggml_tensor_dump("RMSNorm input", x);

        ggml_tensor_t *x2;
        x2 = ggml_mul(ctx, x, x);
        x2 = ggml_mean_ext(ctx, x2, 0/*dim*/);
        x2 = ggml_add_constant(ctx, x2, 1e-6);

        ggml_tensor *rrms = ggml_sqrt(ctx, x2);
        ggml_tensor_t *one = ggml_dup(ctx, rrms);
        one = ggml_constant(ctx, one, 1.0f);
        rrms = ggml_div(ctx, one, rrms);

        x = ggml_mul(ctx, x, rrms); // Dot
        x = ggml_mul(ctx, x, scale); // Dot 

        return x;
    }
};

struct QKNorm {
    int dim = 64;

    struct RMSNorm query_norm;
    struct RMSNorm key_norm;

    void create_weight_tensors(struct ggml_context* ctx) {
        query_norm.dim = dim;
        query_norm.create_weight_tensors(ctx);

        key_norm.dim = dim;
        key_norm.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "query_norm.");
        query_norm.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "key_norm.");
        key_norm.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // useless, place holder !!!!!!!!!!!!!!!
        return x;
    }
};


struct SingleStreamBlock {
    const int hidden_size = 1024;
    const int num_heads = 16;
    const int mlp_ratio = 4;
    const int mlp_hidden_dim = 4096; // hidden_size * mlp_ratio
    const int head_dim = 64; // hidden_size // num_heads

    struct Linear linear1;
    struct Linear linear2;
    struct QKNorm norm;
    struct LayerNorm pre_norm;
    struct SingleModulation modulation;

    void create_weight_tensors(struct ggml_context* ctx) {
        linear1.in_features = hidden_size;
        linear1.out_features = hidden_size * 3 + mlp_hidden_dim;
        linear1.has_bias = true; 
        linear1.create_weight_tensors(ctx);

        linear2.in_features = hidden_size + mlp_hidden_dim;
        linear2.out_features = hidden_size;
        linear2.has_bias = true;
        linear2.create_weight_tensors(ctx);

        norm.dim = head_dim;
        norm.create_weight_tensors(ctx);

        pre_norm.normalized_shape = hidden_size;
        pre_norm.eps = 1e-6;
        pre_norm.elementwise_affine = false;
        pre_norm.create_weight_tensors(ctx);

        // modulation.dim = hidden_size;
        modulation.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "linear1.");
        linear1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "linear2.");
        linear2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
        norm.setup_weight_names(s);
        // snprintf(s, sizeof(s), "%s%s", prefix, "pre_norm.");
        // pre_norm.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "modulation.");
        modulation.setup_weight_names(s);
    }


    // def forward(self, x, vec):
    //     # tensor [x] size: [2, 1874, 1024], min: -257.620056, max: 3559.283691, mean: 0.296953
    //     # tensor [vec] size: [2, 1024], min: -0.274489, max: 5.098103, mean: 0.0289

    //     mod_shift, mod_scale, mod_gate = self.modulation(vec)

    //     x_mod = (mod_scale + 1.0) * self.pre_norm(x) + mod_shift
    //     # x_mod.size() -- [2, 1874, 1024]

    //     # self.linear1(x_mod).size() -- [2, 1874, 7168]
    //     qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

    //     # tensor [qkv] size: [2, 1874, 3072], min: -20.578125, max: 22.84375, mean: 0.003015
    //     q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)  # self.num_heads == 16
    //     # q, k, v is tuple: len = 3
    //     #     tensor [item] size: [2, 16, 1874, 64], min: -12.992188, max: 14.09375, mean: -0.001616
    //     #     tensor [item] size: [2, 16, 1874, 64], min: -17.484375, max: 22.84375, mean: 0.001019
    //     #     tensor [item] size: [2, 16, 1874, 64], min: -20.578125, max: 20.78125, mean: 0.009641
    //     # q, k = self.norm(q, k, v)
    //     q = self.norm.query_norm(q)
    //     k = self.norm.query_norm(k)

    //     # compute attention
    //     attn = attention(q, k, v)
    //     # compute activation in mlp stream, cat again and run second linear layer

    //     # attn.size() -- [2, 1874, 1024]
    //     # self.mlp_act(mlp).size() -- [2, 1874, 4096]
    //     output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))
    //     # output.size() -- [2, 1874, 1024]
        
    //     return x + mod_gate * output


    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, ggml_tensor_t* vec) {
        // x    f32 [1024, 1874, 2, 1], 
        // vec    f32 [1024, 2, 1, 1], 

        // 1. mod_shift, mod_scale, mod_gate = self.modulation(vec)
        std::vector<ggml_tensor *> mod = modulation.forward(ctx, vec); // mod_shift, mod_scale, mod_gate
        ggml_tensor_t *mod_shift = mod[0];
        ggml_tensor_t *mod_scale = mod[1];
        ggml_tensor_t *mod_gate = mod[2];
        // mod_shift    f32 [1024, 1, 2, 1],  (reshaped) (view) (cont)
        // mod_scale    f32 [1024, 1, 2, 1],  (reshaped) (view) (cont)
        // mod_gate    f32 [1024, 1, 2, 1],  (reshaped) (view) (cont)

        // 2. x_mod = (mod_scale + 1.0) * self.pre_norm(x) + mod_shift
        mod_scale = ggml_add_constant(ctx, mod_scale, 1.0f);
        // pre_norm.forward(ctx, x)    f32 [1024, 1882, 2, 1], 
        // mod_scale    f32 [1024, 1, 2, 1], 
        ggml_tensor_t *x_mod = ggml_mul(ctx, pre_norm.forward(ctx, x), mod_scale);
        x_mod = ggml_add(ctx, x_mod, mod_shift);
        // x_mod    f32 [7168, 1874, 2, 1], 

        // 3. qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        x_mod = linear1.forward(ctx, x_mod);
        // x_mod    f32 [7168, 1874, 2, 1], 
        ggml_tensor_t *qkv = ggml_nn_slice(ctx, x_mod, 0/*dim*/, 0/*start*/, 3*hidden_size /*stop*/, 1/*step*/);
        ggml_tensor_t *mlp = ggml_nn_slice(ctx, x_mod, 0/*dim*/, 3*hidden_size /*start*/, 3*hidden_size + mlp_hidden_dim /*stop*/,  1/*step*/);
        // qkv    f32 [3072, 1874, 2, 1],  (view) (cont)
        // mlp    f32 [4096, 1874, 2, 1],  (view) (cont)

        ggml_tensor_t *q = ggml_nn_slice(ctx, qkv, 0/*dim*/, 0*hidden_size/*start*/, 1*hidden_size/*stop*/, 1/*step*/);
        ggml_tensor_t *k = ggml_nn_slice(ctx, qkv, 0/*dim*/, 1*hidden_size/*start*/, 2*hidden_size/*stop*/, 1/*step*/);
        ggml_tensor_t *v = ggml_nn_slice(ctx, qkv, 0/*dim*/, 2*hidden_size/*start*/, 3*hidden_size/*stop*/, 1/*step*/);
        // q    f32 [1024, 1874, 2, 1],  (view) (cont) (view) (cont)
        // k    f32 [1024, 1874, 2, 1],  (view) (cont) (view) (cont)
        // v    f32 [1024, 1874, 2, 1],  (view) (cont) (view) (cont)

        q = ggml_reshape_4d(ctx, q, head_dim, num_heads, q->ne[1], 2); // [1024, 1874, 2] ==> [64, 16, 1874, 2] ==> [64. 1874, 16, 2]
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_reshape_4d(ctx, k, head_dim, num_heads, k->ne[1], 2); // [1024, 1874, 2] ==> [64, 16, 1874, 2] ==> [64. 1874, 16, 2]
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_reshape_4d(ctx, v, head_dim, num_heads, v->ne[1], 2); // [1024, 1874, 2] ==> [64, 16, 1874, 2] ==> [64. 1874, 16, 2]
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        // q    f32 [64, 1874, 16, 2],  (view) (cont) (view) (cont) (reshaped) (permuted) (cont)
        // k    f32 [64, 1874, 16, 2],  (view) (cont) (view) (cont) (reshaped) (permuted) (cont)
        // v    f32 [64, 1874, 16, 2],  (view) (cont) (view) (cont) (reshaped) (permuted) (cont)

        // 5.
        // q = self.norm.query_norm(q)
        // k = self.norm.query_norm(k)
        q = norm.query_norm.forward(ctx, q);
        k = norm.key_norm.forward(ctx, k);
        ggml_tensor_t *attn = attention(ctx, q, k, v);
        // attn    f32 [1024, 1882, 2, 1],  (permuted) (cont) (reshaped)

        // 6. output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))
        mlp = ggml_gelu(ctx, mlp);
        ggml_tensor_t *out = ggml_concat(ctx, attn, mlp, 0/*dim*/);
        out = linear2.forward(ctx, out);
        // out    f32 [1024, 1874, 2, 1], 

        // 7. x = x + mod_gate * output
        // ------------------------------------------------------------------------
        // mod_gate    f32 [1024, 1, 2, 1],  (reshaped) (view) (cont)
        // out    f32 [1024, 1882, 2, 1], 
        out = ggml_mul(ctx, out, mod_gate); // Dot
        x = ggml_add(ctx, x, out);
        // x    f32 [1024, 1874, 2, 1],

    	return x;
    }
};


struct SelfAttention {
    const int dim = 1024;
    const int num_heads = 16;

    struct Linear qkv;
    struct QKNorm norm;
    struct Linear proj;

    void create_weight_tensors(struct ggml_context* ctx) {
        qkv.in_features = dim;
        qkv.out_features = dim * 3;
        qkv.has_bias = true; 
        qkv.create_weight_tensors(ctx);

        norm.dim = dim/num_heads;
        norm.create_weight_tensors(ctx);

        proj.in_features = dim;
        proj.out_features = dim;
        proj.has_bias = true; 
        proj.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "qkv.");
        qkv.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
        norm.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "proj.");
        proj.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // useless, place holder !!!!!!!!!!!!!!!
    	return x;
    }
};


struct DoubleModulation {
    const int dim = 1024;
    const int multiplier = 6;

    struct Linear lin;

    void create_weight_tensors(struct ggml_context* ctx) {
        lin.in_features = dim;
        lin.out_features = multiplier * dim;
        lin.has_bias = true;
        lin.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "lin.");
        lin.setup_weight_names(s);
    }

    // def forward(self, vec):
    //     out = F.silu(vec)

    //     # tensor [out] size: [2, 1024], min: -0.118526, max: 5.067151, mean: 0.023684
    //     out = self.lin(out)
    //     # tensor [out] size: [2, 6144], min: -4.887924, max: 9.55643, mean: -0.091222

    //     out = out[:, None, :] # size() -- [2, 1, 6144]
    //     out = out.chunk(self.multiplier, dim=-1)
    //     # len(out) == 6, ==> out[0].size() -- [2, 1, 1024]
    //     return out[0], out[1], out[2], out[3], out[4], out[5] # shift, scale, gate; shift2, scale2, gate2

    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        ggml_tensor_t *out = ggml_silu(ctx, x);
        out = lin.forward(ctx, out);

        int C0 = (int)out->ne[0];
        int C1 = (int)out->ne[1];
        out = ggml_reshape_3d(ctx, out, C0, 1, C1);

        return ggml_nn_chunks(ctx, out, 0/*dim*/, multiplier); // shift, scale, gate; shift2, scale2, gate2
    }
};


struct DoubleStreamBlock {
    const int hidden_size = 1024;
    const int num_heads = 16;
    const int mlp_ratio = 4;
    const int mlp_hidden_dim = 4096; // hidden_size * mlp_ratio

    struct DoubleModulation img_mod;
    struct LayerNorm img_norm1;
    struct SelfAttention img_attn;
    struct LayerNorm img_norm2;
    struct Linear img_mlp_0;
    struct Linear img_mlp_2;
    // -------------------------------------
    struct DoubleModulation txt_mod;
    struct LayerNorm txt_norm1;
    struct SelfAttention txt_attn;
    struct LayerNorm txt_norm2;
    struct Linear txt_mlp_0;
    struct Linear txt_mlp_2;

    void create_weight_tensors(struct ggml_context* ctx) {
        // img_mod.dim = hidden_size;
        img_mod.create_weight_tensors(ctx);

        img_norm1.normalized_shape = hidden_size;
        img_norm1.eps = 1e-6; // Fixed default values
        img_norm1.elementwise_affine = false;
        img_norm1.create_weight_tensors(ctx);

        // img_attn.dim = hidden_size;
        // img_attn.num_heads = num_heads;
        img_attn.create_weight_tensors(ctx);

        img_norm2.normalized_shape = hidden_size;
        img_norm2.eps = 1e-6; // Fixed default values
        img_norm2.elementwise_affine = false;
        img_norm2.create_weight_tensors(ctx);

        img_mlp_0.in_features = hidden_size;
        img_mlp_0.out_features = mlp_hidden_dim;
        img_mlp_0.has_bias = true; // Fixed default
        img_mlp_0.create_weight_tensors(ctx);

        img_mlp_2.in_features = mlp_hidden_dim;
        img_mlp_2.out_features = hidden_size;
        img_mlp_2.has_bias = true; // Fixed default
        img_mlp_2.create_weight_tensors(ctx);

        // -----------------------------------------------------------------
        txt_mod.create_weight_tensors(ctx);

        txt_norm1.normalized_shape = hidden_size;
        txt_norm1.eps = 1e-6; // Fixed default values
        txt_norm1.elementwise_affine = false;
        txt_norm1.create_weight_tensors(ctx);

        // txt_attn.dim = hidden_size;
        // txt_attn.num_heads = num_heads;
        txt_attn.create_weight_tensors(ctx);

        txt_norm2.normalized_shape = hidden_size;
        txt_norm2.eps = 1e-6; // Fixed default values
        txt_norm2.elementwise_affine = false;
        txt_norm2.create_weight_tensors(ctx);

        txt_mlp_0.in_features = hidden_size;
        txt_mlp_0.out_features = mlp_hidden_dim;
        txt_mlp_0.has_bias = true; // Fixed default
        txt_mlp_0.create_weight_tensors(ctx);

        txt_mlp_2.in_features = mlp_hidden_dim;
        txt_mlp_2.out_features = hidden_size;
        txt_mlp_2.has_bias = true; // Fixed default        
        txt_mlp_2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "img_mod.");
        img_mod.setup_weight_names(s);
        // snprintf(s, sizeof(s), "%s%s", prefix, "img_norm1.");
        // img_norm1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "img_attn.");
        img_attn.setup_weight_names(s);
        // snprintf(s, sizeof(s), "%s%s", prefix, "img_norm2.");
        // img_norm2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "img_mlp.0.");
        img_mlp_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "img_mlp.2.");
        img_mlp_2.setup_weight_names(s);
        // -----------------------------------------------------------------
        snprintf(s, sizeof(s), "%s%s", prefix, "txt_mod.");
        txt_mod.setup_weight_names(s);
        // snprintf(s, sizeof(s), "%s%s", prefix, "txt_norm1.");
        // txt_norm1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "txt_attn.");
        txt_attn.setup_weight_names(s);
        // snprintf(s, sizeof(s), "%s%s", prefix, "txt_norm2.");
        // txt_norm2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "txt_mlp.0.");
        txt_mlp_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "txt_mlp.2.");
        txt_mlp_2.setup_weight_names(s);
    }

    // def forward(self, img, txt, vec):
    //     # tensor [img] size: [2, 512, 1024], min: -7.316819, max: 7.774179, mean: -0.001382
    //     # tensor [txt] size: [2, 1370, 1024], min: -143.35434, max: 155.169907, mean: -0.016531
    //     # -------------------------------------------------------------------------------------
    //     # tensor [vec] size: [2, 1024], min: -0.274489, max: 5.098103, mean: 0.0289

    //     #shift, scale, gate
    //     img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = self.img_mod(vec)
    //     txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = self.txt_mod(vec)

    //     img_modulated = self.img_norm1(img)
    //     img_modulated = (img_mod1_scale + 1.0) * img_modulated + img_mod1_shift

    //     img_qkv = self.img_attn.qkv(img_modulated)

    //     # self.num_heads -- 16
    //     # tensor [img_qkv] size: [2, 512, 3072], min: -6.292969, max: 5.761719, mean: 0.01594
    //     img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    //     # img_q, img_k, img_v is tuple: len = 3
    //     #     tensor [img_q] size: [2, 16, 512, 64], min: -24.125, max: 23.796875, mean: 0.132935
    //     #     tensor [img_k] size: [2, 16, 512, 64], min: -28.15625, max: 28.09375, mean: 0.04367
    //     #     tensor [img_v] size: [2, 16, 512, 64], min: -21.53125, max: 24.40625, mean: 0.032035

    //     # img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
    //     img_q = self.img_attn.norm.query_norm(img_q)
    //     img_k = self.img_attn.norm.key_norm(img_k)

    //     txt_modulated = self.txt_norm1(txt)
    //     txt_modulated = (txt_mod1_scale + 1.0) * txt_modulated + txt_mod1_shift
    //     txt_qkv = self.txt_attn.qkv(txt_modulated)
    //     # tensor [txt_qkv] size: [2, 1370, 3072], min: -22.920546, max: 28.528749, mean: 0.004023

    //     # 3070 = 3 * 16 * 64
    //     txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    //     # txt_q, txt_k, txt_v is tuple: len = 3
    //     #     tensor [item] size: [2, 16, 1370, 64], min: -22.920546, max: 20.372944, mean: 0.026875
    //     #     tensor [item] size: [2, 16, 1370, 64], min: -14.186515, max: 28.528749, mean: -0.022128
    //     #     tensor [item] size: [2, 16, 1370, 64], min: -13.220157, max: 13.752945, mean: 0.007324

    //     # txt_q2 = txt_qkv[:, :, 0000:1024] # [2, 1370, 1024] ==> [2, 1370, 16, 64] ==> [2, 16, 1370, 64]
    //     # txt_k2 = txt_qkv[:, :, 1024:2048]
    //     # txt_v2 = txt_qkv[:, :, 2048:3072]

    //     # txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
    //     txt_q = self.txt_attn.norm.query_norm(txt_q)
    //     txt_k = self.txt_attn.norm.key_norm(txt_k)

    // --------------------------------------------------------------------------------------
    //     q = torch.cat((txt_q, img_q), dim=2) # [2, 16, 1370 + 512, 64] ==== [2, 16, 1882, 64]
    //     k = torch.cat((txt_k, img_k), dim=2)
    //     v = torch.cat((txt_v, img_v), dim=2)

    //     attn = attention(q, k, v)
    //     # tensor [attn] size: [2, 1882, 1024], min: -7.287106, max: 8.247703, mean: 0.012221



    //     # txt.shape -- [2, 1370, 1024]
    //     txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]
    //     # txt_attn.size() -- [2, 1370, 1024]
    //     # img_attn.size() -- [2, 512, 1024]

    //     img = img + img_mod1_gate * self.img_attn.proj(img_attn)
    //     img = img + img_mod2_gate * self.img_mlp((img_mod2_scale + 1.0) * self.img_norm2(img) + img_mod2_shift)

    //     txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
    //     txt = txt + txt_mod2_gate * self.txt_mlp((txt_mod2_scale + 1.0) * self.txt_norm2(txt) + txt_mod2_shift)

    //     # tensor [img] size: [2, 512, 1024], min: -37.988338, max: 7.773553, mean: -0.055882
    //     # tensor [txt] size: [2, 1370, 1024], min: -200.23848, max: 215.347946, mean: -0.12639
    //     assert img.size(1) == 512
    //     # assert txt.size(1) == 1370

    //     # return img, txt # torch.cat((img, txt), dim=1) #==> [2, 1882, 1024]
    //     return torch.cat((img, txt), dim=1) # [2, 1882, 1024]

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* img, ggml_tensor_t* txt, ggml_tensor_t* vec) {
        // tensor [img] size: [2, 512, 1024], min: -7.316819, max: 7.774179, mean: -0.001382
        // tensor [txt] size: [2, 1370, 1024], min: -143.35434, max: 155.169907, mean: -0.016531
        // tensor [vec] size: [2, 1024], min: -0.274489, max: 5.098103, mean: 0.0289

        // shift, scale, gate
        // 1. img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = self.img_mod(vec)
        std::vector<ggml_tensor_t *> img_mods = img_mod.forward(ctx, vec);
        ggml_tensor_t *img_mod1_shift = img_mods[0];
        ggml_tensor_t *img_mod1_scale = img_mods[1];
        ggml_tensor_t *img_mod1_gate = img_mods[2];
        ggml_tensor_t *img_mod2_shift = img_mods[3];
        ggml_tensor_t *img_mod2_scale = img_mods[4];
        ggml_tensor_t *img_mod2_gate =  img_mods[5];
        // img_mod1_shift    f32 [1024, 1, 2, 1],  (reshaped) (view) (cont)
        // img_mod1_scale    f32 [1024, 1, 2, 1],  (reshaped) (view) (cont)
        // img_mod1_gate    f32 [1024, 1, 2, 1],  (reshaped) (view) (cont)
        // img_mod2_shift    f32 [1024, 1, 2, 1],  (reshaped) (view) (cont)
        // img_mod2_scale    f32 [1024, 1, 2, 1],  (reshaped) (view) (cont)
        // img_mod2_gate    f32 [1024, 1, 2, 1],  (reshaped) (view) (cont)

        // ----------------------------------------------------------------
        // 2. txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = self.txt_mod(vec)        
        std::vector<ggml_tensor_t *> txt_mods = txt_mod.forward(ctx, vec);
        ggml_tensor_t *txt_mod1_shift = txt_mods[0];
        ggml_tensor_t *txt_mod1_scale = txt_mods[1];
        ggml_tensor_t *txt_mod1_gate = txt_mods[2];
        ggml_tensor_t *txt_mod2_shift = txt_mods[3];
        ggml_tensor_t *txt_mod2_scale = txt_mods[4];
        ggml_tensor_t *txt_mod2_gate = txt_mods[5];
        // ----------------------------------------------------------------

        // 3.  img_modulated = self.img_norm1(img)
        //     img_modulated = (img_mod1_scale + 1.0) * img_modulated + img_mod1_shift        
        ggml_tensor_t *img_modulated = img_norm1.forward(ctx, img);
        img_mod1_scale = ggml_add_constant(ctx, img_mod1_scale, 1.0f);
        // img_mod1_scale    f32 [1024, 1, 2, 1], 
        // img_modulated1    f32 [1024, 512, 2, 1], 

        img_modulated = ggml_mul(ctx, img_modulated, img_mod1_scale); // Dot Product


        // 4.  img_qkv = self.img_attn.qkv(img_modulated)
        ggml_tensor_t *img_qkv = img_attn.qkv.forward(ctx, img_modulated);
        // img_qkv    f32 [3072, 512, 2, 1], 

        ggml_tensor_t *img_q = ggml_nn_slice(ctx, img_qkv, 0/*dim*/, 0*hidden_size/*start*/, 1*hidden_size/*stop*/, 1/*step*/);
        ggml_tensor_t *img_k = ggml_nn_slice(ctx, img_qkv, 0/*dim*/, 1*hidden_size/*start*/, 2*hidden_size/*stop*/, 1/*step*/);
        ggml_tensor_t *img_v = ggml_nn_slice(ctx, img_qkv, 0/*dim*/, 2*hidden_size/*start*/, 3*hidden_size/*stop*/, 1/*step*/);

        // const int num_heads = 16;
        // [1024, 512, 2] ==> [64, 16, 512, 2] ==> [64, 512, 16, 2]
        img_q = ggml_reshape_4d(ctx, img_q, (int)img_q->ne[0]/num_heads, num_heads, (int)img_q->ne[1], (int)img_q->ne[2]);
        img_k = ggml_reshape_4d(ctx, img_k, (int)img_k->ne[0]/num_heads, num_heads, (int)img_k->ne[1], (int)img_k->ne[2]);
        img_v = ggml_reshape_4d(ctx, img_v, (int)img_v->ne[0]/num_heads, num_heads, (int)img_v->ne[1], (int)img_v->ne[2]);
        img_q = ggml_cont(ctx, ggml_permute(ctx, img_q, 0, 2, 1, 3)); // [64, 16, 512, 2] ==> [64, 512, 16, 2]
        img_k = ggml_cont(ctx, ggml_permute(ctx, img_k, 0, 2, 1, 3)); // [64, 16, 512, 2] ==> [64, 512, 16, 2]
        img_v = ggml_cont(ctx, ggml_permute(ctx, img_v, 0, 2, 1, 3)); // [64, 16, 512, 2] ==> [64, 512, 16, 2]
        // -----------------------------------------------------------------------------

        img_q = img_attn.norm.query_norm.forward(ctx, img_q);
        img_k = img_attn.norm.key_norm.forward(ctx, img_k);
        // -----------------------------------------------------------------------------


        ggml_tensor_t *txt_modulated = txt_norm1.forward(ctx, txt);
        txt_mod1_scale = ggml_add_constant(ctx, txt_mod1_scale, 1.0f);
        txt_modulated = ggml_mul(ctx, txt_modulated, txt_mod1_scale); // Dot Product

        txt_modulated = ggml_add(ctx, txt_modulated, txt_mod1_shift);
        ggml_tensor_t *txt_qkv = txt_attn.qkv.forward(ctx, txt_modulated);
        ggml_tensor_t *txt_q = ggml_nn_slice(ctx, txt_qkv, 0/*dim*/, 0*hidden_size/*start*/, 1*hidden_size/*stop*/, 1/*step*/);
        ggml_tensor_t *txt_k = ggml_nn_slice(ctx, txt_qkv, 0/*dim*/, 1*hidden_size/*start*/, 2*hidden_size/*stop*/, 1/*step*/);
        ggml_tensor_t *txt_v = ggml_nn_slice(ctx, txt_qkv, 0/*dim*/, 2*hidden_size/*start*/, 3*hidden_size/*stop*/, 1/*step*/);
        txt_q = ggml_reshape_4d(ctx, txt_q, (int)txt_q->ne[0]/num_heads, num_heads, (int)txt_q->ne[1], (int)txt_q->ne[2]);
        txt_k = ggml_reshape_4d(ctx, txt_k, (int)txt_k->ne[0]/num_heads, num_heads, (int)txt_k->ne[1], (int)txt_k->ne[2]);
        txt_v = ggml_reshape_4d(ctx, txt_v, (int)txt_v->ne[0]/num_heads, num_heads, (int)txt_v->ne[1], (int)txt_v->ne[2]);
        txt_q = ggml_cont(ctx, ggml_permute(ctx, txt_q, 0, 2, 1, 3)); // [64, 16, 1370, 2] ==> [64, 1370, 16, 2]
        txt_k = ggml_cont(ctx, ggml_permute(ctx, txt_k, 0, 2, 1, 3)); // [64, 16, 1370, 2] ==> [64, 1370, 16, 2]
        txt_v = ggml_cont(ctx, ggml_permute(ctx, txt_v, 0, 2, 1, 3)); // [64, 16, 1370, 2] ==> [64, 1370, 16, 2]
        // txt_q    f32 [64, 1370, 16, 2],  (view) (cont) (reshaped) (permuted) (cont)
        // txt_k    f32 [64, 1370, 16, 2],  (view) (cont) (reshaped) (permuted) (cont)
        // txt_v    f32 [64, 1370, 16, 2],  (view) (cont) (reshaped) (permuted) (cont)

        txt_q = txt_attn.norm.query_norm.forward(ctx, txt_q);
        txt_k = txt_attn.norm.key_norm.forward(ctx, txt_k);

        ggml_tensor_t *q = ggml_concat(ctx, txt_q, img_q, 1/*dim*/);
        ggml_tensor_t *k = ggml_concat(ctx, txt_k, img_k, 1/*dim*/);
        ggml_tensor_t *v = ggml_concat(ctx, txt_v, img_v, 1/*dim*/);
        ggml_tensor_t *attn = attention(ctx, q, k, v);
        // ggml_tensor_dump("===> attn 10", attn);
        // ===> attn 10    f32 [1024, 1882, 2, 1],  (permuted) (cont) (reshaped)

        int S = 512;
        ggml_tensor_t *img_attn2 = ggml_nn_slice(ctx, attn, 1/*dim*/, 0/*start*/, S/*stop*/, 1/*step*/);
        ggml_tensor_t *txt_attn2 = ggml_nn_slice(ctx, attn, 1/*dim*/, S/*start*/, attn->ne[1]/*stop*/, 1/*step*/);
        // img_attn2    f32 [1024, 512, 2, 1],  (permuted) (cont) (reshaped) (view) (cont)
        // txt_attn2    f32 [1024, 1370, 2, 1],  (permuted) (cont) (reshaped) (view) (cont)

        // img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        // -----------------------------------------------------------------------------------------------------
        // img_attn.forward(ctx, img_attn2)    f32 [1024, 512, 2, 1],  (permuted) (cont) (reshaped) (view) (cont)
        // img_mod1_gate    f32 [1024, 1, 2, 1],  (reshaped) (view) (cont)
        img_mod1_gate = ggml_mul(ctx, img_attn.forward(ctx, img_attn2), img_mod1_gate); // Dot
        img = ggml_add(ctx, img, img_mod1_gate);

        // img = img + img_mod2_gate * self.img_mlp((img_mod2_scale + 1.0) * self.img_norm2(img) + img_mod2_shift)
        // -----------------------------------------------------------------------------------------------------
        img_mod2_scale = ggml_add_constant(ctx, img_mod2_scale, 1.0f);
        // img_norm2.forward(ctx, img)    f32 [1024, 512, 2, 1], 
        // img_mod2_scale    f32 [1024, 1, 2, 1], 
        img_mod2_scale = ggml_mul(ctx, img_norm2.forward(ctx, img), img_mod2_scale); // Dot

        img_mod2_scale = ggml_add(ctx, img_mod2_scale, img_mod2_shift);

        img_mod2_scale = img_mlp_0.forward(ctx, img_mod2_scale);
        img_mod2_scale = ggml_gelu(ctx, img_mod2_scale);
        img_mod2_scale = img_mlp_2.forward(ctx, img_mod2_scale);

        // img_mod2_scale    f32 [1024, 512, 2, 1], 
        // img_mod2_gate    f32 [1024, 1, 2, 1],  (reshaped) (view) (cont)
        img_mod2_gate = ggml_mul(ctx, img_mod2_scale, img_mod2_gate); // Dot
        img = ggml_add(ctx, img, img_mod2_gate);

        // txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        // -----------------------------------------------------------------------------------------------------
        txt_attn2 = txt_attn.proj.forward(ctx, txt_attn2);
        // txt_mod1_gate    f32 [1024, 1, 2, 1],  (reshaped) (view) (cont)
        // txt_attn2    f32 [1024, 1370, 2, 1], 
        txt_mod1_gate = ggml_mul(ctx, txt_attn2, txt_mod1_gate); // Dot
        txt = ggml_add(ctx, txt, txt_mod1_gate);


        // txt = txt + txt_mod2_gate * self.txt_mlp((txt_mod2_scale + 1.0) * self.txt_norm2(txt) + txt_mod2_shift)
        // -----------------------------------------------------------------------------------------------------
        txt_mod2_scale = ggml_add_constant(ctx, txt_mod2_scale, 1.0f);
        // txt_mod2_scale    f32 [1024, 1, 2, 1], 
        // txt_norm2.forward(ctx, txt)    f32 [1024, 1370, 2, 1],
        txt_mod2_scale = ggml_mul(ctx, txt_norm2.forward(ctx, txt), txt_mod2_scale); // Dot
        txt_mod2_scale = ggml_add(ctx, txt_mod2_scale, txt_mod2_shift);

        // self.txt_mlp = nn.Sequential(
        //     nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
        //     GELU(approximate="tanh"),
        //     nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        // )
        txt_mod2_scale = txt_mlp_0.forward(ctx, txt_mod2_scale);
        txt_mod2_scale = ggml_gelu(ctx, txt_mod2_scale);
        txt_mod2_scale = txt_mlp_2.forward(ctx, txt_mod2_scale);

        // txt_mod2_gate    f32 [1024, 1, 2, 1],  (reshaped) (view) (cont)
        // txt_mod2_scale    f32 [1024, 1370, 2, 1], 
        txt_mod2_gate = ggml_mul(ctx, txt_mod2_scale, txt_mod2_gate); // Dot
        txt = ggml_add(ctx, txt, txt_mod2_gate);

        // -----------------------------------------------------------------------------------------------------
        // img    f32 [1024, 512, 2, 1], 
        // txt    f32 [1024, 1370, 2, 1], 
        ggml_tensor_t *out = ggml_concat(ctx, img, txt, 1/*dim*/);
        // out    f32 [1024, 1882, 2, 1], 

        return out;
    }
};


struct MLPEmbedder {
    const int in_dim = 256;
    const int hidden_dim = 1024;

    struct Linear in_layer;
    struct Linear out_layer;

    void create_weight_tensors(struct ggml_context* ctx) {
        in_layer.in_features = in_dim;
        in_layer.out_features = hidden_dim;
        in_layer.has_bias = true; 
        in_layer.create_weight_tensors(ctx);

        out_layer.in_features = hidden_dim;
        out_layer.out_features = hidden_dim;
        out_layer.has_bias = true; 
        out_layer.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "in_layer.");
        in_layer.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "out_layer.");
        out_layer.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = in_layer.forward(ctx, x);
        x = ggml_silu(ctx, x);
        x = out_layer.forward(ctx, x);
    	return x;
    }
};

// struct Hunyuan3DDiT
struct DiTNetwork : ggml::GGMLNetwork {
    const int in_channels = 64;
    const int context_in_dim = 1536;
    const int hidden_size = 1024;
    const int mlp_ratio = 4;
    const int num_heads = 16;
    const float time_factor = 1000.0f;

    struct Linear latent_in;
    struct MLPEmbedder time_in;
    struct Linear cond_in;
    struct DoubleStreamBlock double_blocks[8]; // depth
    struct SingleStreamBlock single_blocks[16]; // depth_single_blocks
    struct LastLayer final_layer;

    size_t get_graph_size()
    {
        return 8*GGML_DEFAULT_GRAPH_SIZE; // 2048
    }

    void create_weight_tensors(struct ggml_context* ctx) {
        latent_in.in_features = in_channels;
        latent_in.out_features = hidden_size;
        latent_in.has_bias = true; // Fixed default
        latent_in.create_weight_tensors(ctx);

        // time_in.in_dim = 256;
        // time_in.hidden_dim = 1024; // hidden_size
        time_in.create_weight_tensors(ctx);

        cond_in.in_features = context_in_dim;
        cond_in.out_features = hidden_size;
        cond_in.has_bias = true; // Fixed default
        cond_in.create_weight_tensors(ctx);

        for (int i = 0; i < 8; i++) {
            // double_blocks[i].hidden_size = 1024; // hidden_size
            // double_blocks[i].num_heads = 16; // num_heads
            // double_blocks[i].mlp_ratio = 4; // mlp_ratio
            double_blocks[i].create_weight_tensors(ctx);
        }

        for (int i = 0; i < 16; i++) {
            // single_blocks[i].hidden_size = 1024; // hidden_size
            // single_blocks[i].num_heads = 16; // num_heads
            // single_blocks[i].mlp_ratio = 4; // mlp_ratio
            single_blocks[i].create_weight_tensors(ctx);
        }

        // final_layer.hidden_size = 1024; // hidden_size
        // final_layer.patch_size = 1;
        // final_layer.out_channels = 64; // in_channels
        final_layer.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "latent_in.");
        latent_in.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "time_in.");
        time_in.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "cond_in.");
        cond_in.setup_weight_names(s);

        for (int i = 0; i < 8; i++) {
            snprintf(s, sizeof(s), "%sdouble_blocks.%d.", prefix, i);
            double_blocks[i].setup_weight_names(s);
        }

        for (int i = 0; i < 16; i++) {
            snprintf(s, sizeof(s), "%ssingle_blocks.%d.", prefix, i);
            single_blocks[i].setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "final_layer.");
        final_layer.setup_weight_names(s);
    }

    // def forward(self, x, t, cond):
    //     # tensor [x] size: [2, 512, 64], min: -4.179688, max: 4.238281, mean: 0.005243
    //     # tensor [t] size: [2], min: 0.0, max: 0.0, mean: 0.0
    //     # tensor [cond] size: [2, 1370, 1536], min: -15.28125, max: 14.375, mean: -0.009306
    //     # --------------------------------------------------------------------------------
    //     latent = self.latent_in(x)
    //     # tensor [latent] size: [2, 512, 1024], min: -6.734375, max: 6.480469, mean: -0.002356

    //     # self.time_factor --- 1000
    //     # timestep_embedding(t, 256, self.time_factor).size() -- [2, 256]
    //     vec = self.time_in(timestep_embedding(t, 256, self.time_factor).to(dtype=latent.dtype))
    //     # tensor [vec] size: [2, 1024], min: -0.274414, max: 5.097656, mean: 0.028905

    //     cond = self.cond_in(cond)
    //     # tensor [cond] size: [2, 1370, 1024], min: -143.375, max: 155.125, mean: -0.016774

    //     for block in self.double_blocks:  # len(self.double_blocks) === 8
    //         latent, cond = block(img=latent, txt=cond, vec=vec)

    //     # tensor [cond] size: [2, 1370, 1024], min: -263.0, max: 3560.0, mean: 0.430395
    //     # tensor [latent] size: [2, 512, 1024], min: -93.375, max: 106.9375, mean: -0.064714
    //     latent = torch.cat((cond, latent), dim=1)
    //     # tensor [latent] size: [2, 1882, 1024], min: -263.0, max: 3560.0, mean: 0.2957

    //     for block in self.single_blocks:  # len(self.single_blocks) === 16
    //         latent = block(latent, vec=vec)

    //     # tensor [latent] size: [2, 1882, 1024], min: -207.5, max: 3752.0, mean: 0.260497
    //     latent = latent[:, cond.shape[1] :, ...]
    //     # tensor [latent] size: [2, 512, 1024], min: -64.3125, max: 203.875, mean: -0.013699

    //     latent = self.final_layer(latent, vec)
    //     # tensor [latent] size: [2, 512, 64], min: -4.257812, max: 4.207031, mean: -0.006032

    //     return latent

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_ASSERT(argc == 3);
        ggml_tensor_t *x = argv[0];
        ggml_tensor_t *t = argv[1];
        ggml_tensor_t *cond = argv[2];

        ggml_tensor_t *latent = latent_in.forward(ctx, x);
        // latent    f32 [1024, 512, 2, 1], 

        ggml_tensor_t *vec = timestep_embedding(ctx, t, 256, time_factor);
        vec = time_in.forward(ctx, vec);
        // vec    f32 [1024, 2, 1, 1], 
        cond = cond_in.forward(ctx, cond);
        // cond    f32 [1024, 1370, 2, 1], 

        ggml_tensor_t *latent_cond;
        for (int i = 0; i < 8; i++) {
            latent_cond = double_blocks[i].forward(ctx, latent, cond, vec);
            // latent_cond    f32 [1024, 1882->1874, 2, 1], 
            // latent = latent_cond[:, 0:512, :] # img ...
            // cond = latent_cond[:, 512: -1, :] # // text
            latent = ggml_nn_slice(ctx, latent_cond, 1/*dim*/, 0/*start*/, 512/*stop*/, 1/*step*/);
            cond = ggml_nn_slice(ctx, latent_cond, 1/*dim*/, 512/*start*/, (int)latent_cond->ne[1] - 1/*stop*/, 1/*step*/);
        }
        // cond     f32 [1024, 1362, 2, 1],  (view) (cont)
        // latent     f32 [1024, 512, 2, 1],  (view) (cont)
        latent = ggml_concat(ctx, cond, latent, 1/*dim*/);
        // latent1    f32 [1024, 1874, 2, 1], 

        for (int i = 0; i < 16; i++) {
            latent = single_blocks[i].forward(ctx, latent, vec);
        }
        // latent    f32 [1024, 1874, 2, 1], 

        latent = ggml_nn_slice(ctx, latent, 1 /*dim*/, (int)cond->ne[1] /*start*/, (int)latent->ne[1] /*stop*/, 1 /*step*/);
        // latent   f32 [1024, 512, 2, 1],  (view) (cont)

        latent = final_layer.forward(ctx, latent, vec);
        // latent    f32 [64, 512, 2, 1], 

    	return latent;
    }
};


// struct DiTModel {
//     DiTNetwork dit_net;
//     ggml::GGMLModel model;

//     int init(int device)
//     {
//         dit_net.set_device(device);
//         dit_net.start_engine();
//         dit_net.dump();

//         check_point(model.preload("models/image3d_shape.gguf") == RET_OK);

//         dit_net.load_weight(&model, "shape_dit.");
//         model.clear();

//         return RET_OK;
//     }

//     TENSOR* forward(TENSOR* x, TENSOR* t, TENSOR* cond)
//     {
//         TENSOR* argv[3];
//         argv[0] = x;
//         argv[1] = t;
//         argv[2] = cond;

//         // tensor [x] size: [2, 512, 64], min: -4.179688, max: 4.238281, mean: 0.005243
//         // tensor [t] size: [2], min: 0.0, max: 0.0, mean: 0.0
//         // tensor [cond] size: [2, 1370, 1536], min: -15.28125, max: 14.375, mean: -0.009306
//         TENSOR* y = dit_net.engine_forward(ARRAY_SIZE(argv), argv);
//         // tensor [y] size: [2, 512, 64], min: -4.257812, max: 4.207031, mean: -0.006032

//         return y;
//     }

//     void exit()
//     {
//         dit_net.stop_engine();
//     }
// };

#endif // __HUNYUAN3DDIT__H__
