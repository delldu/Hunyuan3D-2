#ifndef __SHAPEVAE__H__
#define __SHAPEVAE__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"

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

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

    	return x;
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
        x = ggml_gelu(ctx, x);
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
        frequencies = ggml_arange(ctx, 0.0f, (float)num_freqs, 1.0f);
        frequencies = ggml_scale(ctx, frequencies, 2.0);
    }

    void setup_weight_names(const char *prefix) {
        // char s[GGML_MAX_NAME];
        GGML_UNUSED(prefix);        
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

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

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

    	return x;
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
        latents = ggml_scale(ctx, latents, scale_factor);
        latents = post_kl.forward(ctx, latents);
        latents = transformer.forward(ctx, latents);


    	return latents;
    }
};

#endif // __SHAPEVAE__H__
