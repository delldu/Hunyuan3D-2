#ifndef __HUNYUAN3DDIT__H__
#define __HUNYUAN3DDIT__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"

struct LastLayer {
    const int hidden_size = 1024;
    const int patch_size = 1;
    const int out_channels = 64;

    // network params
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

    // def forward(self, x, vec):
    //     # tensor [vec] size: [2, 1024], min: -0.274489, max: 5.098103, mean: 0.0289
    //     shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
    //     # tensor [shift] size: [2, 1024], min: -0.517993, max: 0.572035, mean: 0.006371
    //     # tensor [scale] size: [2, 1024], min: -6.589389, max: 3.878436, mean: -1.043143

    //     x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]

    //     x = self.linear(x)
    //     return x
    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, ggml_tensor_t* vec) {
        vec = ggml_silu(ctx, vec);
        vec = adaLN_modulation_1.forward(ctx, vec);

        ggml_tensor_dump("==> vec", vec);
        ggml_tensor_t *shift = ggml_nn_slice(ctx, vec, 0/*dim*/, 0 /*start*/, hidden_size, 1/*step*/);
        ggml_tensor_t *scale = ggml_nn_slice(ctx, vec, 0/*dim*/, hidden_size /*start*/, 2*hidden_size, 1/*step*/);

        int C0 = (int)shift->ne[0];
        int C1 = (int)shift->ne[1];
        shift = ggml_reshape_3d(ctx, shift, C0, 1, C1);
        scale = ggml_reshape_3d(ctx, scale, C0, 1, C1);

        scale = ggml_add_constant(ctx, scale, 1.0f);
        x = norm_final.forward(ctx, x);


        ggml_tensor_dump("==> shift", shift);
        ggml_tensor_dump("==> scale", scale);


        x = ggml_nn_mul_mat(ctx, scale, x);
        x = ggml_add(ctx, x, shift);

        x = linear.forward(ctx, x);
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

        ggml_tensor_dump("SingleModulation", out);

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

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = ggml_mul(ctx, x, x);
        x = ggml_mean_ext(ctx, x, 0/*dim*/);
        x = ggml_add_constant(ctx, x, 1e-6);

        ggml_tensor *rrms = ggml_sqrt(ctx, x);
        ggml_tensor_t *one = ggml_dup(ctx, rrms);
        one = ggml_constant(ctx, one, 1.0f);
        rrms = ggml_div(ctx, one, rrms);

        x = ggml_nn_mul_mat(ctx, x, rrms);
        x = ggml_nn_mul_mat(ctx, x, scale);
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
        // # !!!!!!!!!!!!!!! useless, place holder ...
        return x;
    }
};


struct SingleStreamBlock {
    const int hidden_size = 1024;
    const int num_heads = 16;
    const int mlp_ratio = 4;
    const int mlp_hidden_dim = 4096; // hidden_size * mlp_ratio
    const int head_dim = 64; // hidden_size // num_heads

    // network params
    struct Linear linear1;
    struct Linear linear2;
    struct QKNorm norm;
    struct LayerNorm pre_norm;
    // struct GELU mlp_act;
    struct SingleModulation modulation;

    void create_weight_tensors(struct ggml_context* ctx) {
        linear1.in_features = hidden_size;
        linear1.out_features = hidden_size * 3 + mlp_hidden_dim;
        linear1.has_bias = true; // Fixed default
        linear1.create_weight_tensors(ctx);

        linear2.in_features = hidden_size + mlp_hidden_dim;
        linear2.out_features = hidden_size;
        linear2.has_bias = true; // Fixed default
        linear2.create_weight_tensors(ctx);

        norm.dim = head_dim;
        norm.create_weight_tensors(ctx);

        pre_norm.normalized_shape = hidden_size;
        pre_norm.eps = 1e-6; // Fixed default values
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

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, ggml_tensor_t* vec) {
    	// please implement forward by your self, please !!!

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
        qkv.has_bias = true; // Fixed default
        qkv.create_weight_tensors(ctx);

        norm.dim = dim/num_heads;
        norm.create_weight_tensors(ctx);

        proj.in_features = dim;
        proj.out_features = dim;
        proj.has_bias = true; // Fixed default
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
    	// useless !!!
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
        lin.has_bias = true; // Fixed default
        lin.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "lin.");
        lin.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};


struct DoubleStreamBlock {
    const int hidden_size = 1024;
    const int num_heads = 16;
    const int mlp_ratio = 4;
    const int mlp_hidden_dim = 4096; // hidden_size * mlp_ratio

    // network params
    struct DoubleModulation img_mod;
    struct LayerNorm img_norm1;
    struct SelfAttention img_attn;
    // struct GELU img_mlp_1;
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
        snprintf(s, sizeof(s), "%s%s", prefix, "img_mlp_0.");
        img_mlp_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "img_mlp_2.");
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

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

    	return x;
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
        in_layer.has_bias = true; // Fixed default
        in_layer.create_weight_tensors(ctx);

        out_layer.in_features = hidden_dim;
        out_layer.out_features = hidden_dim;
        out_layer.has_bias = true; // Fixed default
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

struct Hunyuan3DDiT {
    const int in_channels = 64;
    const int context_in_dim = 1536;
    const int hidden_size = 1024;
    const int mlp_ratio = 4;
    const int num_heads = 16;
    // const int depth = 8;
    // const int depth_single_blocks = 16;
    const int time_factor = 1000;

    // network params
    struct Linear latent_in;
    struct MLPEmbedder time_in;
    struct Linear cond_in;

    struct DoubleStreamBlock double_blocks[8]; // depth

    struct SingleStreamBlock single_blocks[16]; // depth_single_blocks
    struct LastLayer final_layer;

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
            snprintf(s, sizeof(s), "%single_blocks.%d.", prefix, i);
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

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, ggml_tensor_t* t, ggml_tensor_t* cond) {
        ggml_tensor_t *latent = latent_in.forward(ctx, x);
        ggml_tensor_t *vec; //  = timestep_embedding.forward(ctx, t, 256, time_factor);
        vec = time_in.forward(ctx, vec);
        cond = cond_in.forward(ctx, cond);

        // for block in self.double_blocks:  # len(self.double_blocks) === 8
        //     latent, cond = block(img=latent, txt=cond, vec=vec)
        latent = ggml_concat(ctx, cond, latent, 1/*dim*/);

        for (int i = 0; i < 16; i++) {
            latent = single_blocks[i].forward(ctx, latent, vec);
        }

        int S = (int)cond->ne[1];
        latent = ggml_nn_slice(ctx, latent, 1 /*dim*/, 0 /*start*/, S /*stop*/, 1 /*step*/);

        latent = final_layer.forward(ctx, latent, vec);

    	return latent;
    }
};

#endif // __HUNYUAN3DDIT__H__
