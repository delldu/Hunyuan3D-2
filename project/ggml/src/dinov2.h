#ifndef __DINOV2__H__
#define __DINOV2__H__
// #include "ggml_engine.h"
#include "ggml_model.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"


ggml_tensor_t* scaled_dot_product_attention(struct ggml_context* ctx, 
    ggml_tensor_t* query, ggml_tensor_t *key, ggml_tensor_t *value) {

    float scale = 1.0f/sqrtf((float)query->ne[0]);
    key = ggml_cont(ctx, ggml_permute(ctx, key, 1, 0, 2, 3)); // [64, 1370, 24, 1]  --> [1370, 64, 24, 1]

    ggml_tensor_t *attn_weight = ggml_nn_mul_mat(ctx, query, key);
    attn_weight = ggml_scale(ctx, attn_weight, scale);
    attn_weight = ggml_softmax(ctx, attn_weight, 0 /*dim*/);

    ggml_tensor_t *new_value = ggml_nn_mul_mat(ctx, attn_weight, value);

    return new_value; // f32 [64, 1370, 24, 1], 
}

ggml_tensor_t* sdpa_attention_forward(struct ggml_context* ctx, ggml_tensor_t* query, ggml_tensor_t *key, ggml_tensor_t *value) {
    query = ggml_cont(ctx, query);
    key = ggml_cont(ctx, key);
    value = ggml_cont(ctx, value);

    ggml_tensor_t *attn_output = scaled_dot_product_attention(ctx, query, key, value);
    attn_output = ggml_cont(ctx, ggml_permute(ctx, attn_output, 0, 2, 1, 3)); // [64, 1370, 24, 1] --> [64, 24, 1370, 1]
    return attn_output;
}


struct Dinov2SwiGLUFFN {
    const int hidden_size = 1536;
    const int hidden_features = 4096;

    struct Linear weights_in;
    struct Linear weights_out;

    void create_weight_tensors(struct ggml_context* ctx) {
        weights_in.in_features = hidden_size;
        weights_in.out_features = 2*hidden_features;
        weights_in.has_bias = true; // Fixed default
        weights_in.create_weight_tensors(ctx);

        weights_out.in_features = hidden_features;
        weights_out.out_features = hidden_size;
        weights_out.has_bias = true; // Fixed default
        weights_out.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "weights_in.");
        weights_in.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "weights_out.");
        weights_out.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        ggml_tensor_t *h = weights_in.forward(ctx, x);
        // h -- torch.Size([1, 1370, 8192])
        ggml_tensor_t *x1 = ggml_nn_slice(ctx, h, 0/*dim*/, 0 /*start*/, hidden_features /*stop*/, 1 /*step*/);
        ggml_tensor_t *x2 = ggml_nn_slice(ctx, h, 0/*dim*/, hidden_features /*start*/, 2*hidden_features /*stop*/, 1 /*step*/);
        h = ggml_mul(ctx, ggml_silu(ctx, x1), x2);
        h = weights_out.forward(ctx, h);
    	return h;
    }
};


struct Dinov2LayerScale {
    const int hidden_size = 1536;

    ggml_tensor_t* lambda1;
    
    void create_weight_tensors(struct ggml_context* ctx) {
        lambda1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
    }

    void setup_weight_names(const char *prefix) {
        ggml_format_name(lambda1, "%s%s", prefix, "lambda1");
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
    	return ggml_mul(ctx, x, lambda1);
    }
};


struct Dinov2SelfOutput {
    const int hidden_size = 1536;

    struct Linear dense;

    void create_weight_tensors(struct ggml_context* ctx) {
        dense.in_features = hidden_size;
        dense.out_features = hidden_size;
        dense.has_bias = true; // Fixed default
        dense.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "dense.");
        dense.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = dense.forward(ctx, x);
    	return x;
    }
};


struct Dinov2SelfAttention {
    const int hidden_size = 1536;

    const int num_attention_heads = 24;
    const int attention_head_size = 64;
    const int all_head_size = 1536;
    const float scaling = 0.125;

    struct Linear query;
    struct Linear key;
    struct Linear value;

    void create_weight_tensors(struct ggml_context* ctx) {
        query.in_features = hidden_size;
        query.out_features = all_head_size;
        query.has_bias = true; // Fixed default
        query.create_weight_tensors(ctx);

        key.in_features = hidden_size;
        key.out_features = all_head_size;
        key.has_bias = true; // Fixed default
        key.create_weight_tensors(ctx);

        value.in_features = hidden_size;
        value.out_features = all_head_size;
        value.has_bias = true; // Fixed default
        value.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "query.");
        query.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "key.");
        key.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "value.");
        value.setup_weight_names(s);
    }

    ggml_tensor_t* transpose_for_scores(struct ggml_context* ctx, ggml_tensor_t* x) {
        int HW = (int)x->ne[0];
        int C = (int)x->ne[1];
        int B = (int)x->ne[2];

        x = ggml_reshape_4d(ctx, x, attention_head_size, num_attention_heads, C, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        return x;
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        ggml_tensor_t *key_layer = transpose_for_scores(ctx, key.forward(ctx, x));
        ggml_tensor_t *query_layer = transpose_for_scores(ctx, query.forward(ctx, x));
        ggml_tensor_t *value_layer = transpose_for_scores(ctx, value.forward(ctx, x));

        ggml_tensor_t *context_layer = sdpa_attention_forward(ctx, query_layer, key_layer, value_layer);

        int W = (int)context_layer->ne[0];
        int H = (int)context_layer->ne[1];
        int C = (int)context_layer->ne[2];
        int B = (int)context_layer->ne[3];

        // context_layer    f32 [64, 24, 1370, 1]
        context_layer = ggml_reshape_3d(ctx, context_layer, H*W /*==all_head_size*/, C, B);
        // context_layer    f32 [1536, 1370, 1, 1]
    	return context_layer;
    }
};

struct Dinov2Attention {
    struct Dinov2SelfAttention attention;
    struct Dinov2SelfOutput output;

    void create_weight_tensors(struct ggml_context* ctx) {
        attention.create_weight_tensors(ctx);
        output.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "attention.");
        attention.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "output.");
        output.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = attention.forward(ctx, x);
        x = output.forward(ctx, x);
    	return x;
    }
};


struct Dinov2Layer {
    const int hidden_size = 1536;
    
    struct LayerNorm norm1;
    struct Dinov2Attention attention;
    struct Dinov2LayerScale layer_scale1;

    struct LayerNorm norm2;
    struct Dinov2SwiGLUFFN mlp;
    struct Dinov2LayerScale layer_scale2;


    void create_weight_tensors(struct ggml_context* ctx) {
        norm1.normalized_shape = hidden_size;
        norm1.eps = 1e-6;
        norm1.create_weight_tensors(ctx);
        attention.create_weight_tensors(ctx);
        layer_scale1.create_weight_tensors(ctx);

        norm2.normalized_shape = hidden_size;
        norm2.eps = 1e-6;
        norm2.create_weight_tensors(ctx);
        mlp.create_weight_tensors(ctx);
        layer_scale2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "attention.");
        attention.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer_scale1.");
        layer_scale1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.");
        mlp.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer_scale2.");
        layer_scale2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        ggml_tensor_t *attention_output = norm1.forward(ctx, x);
        attention_output = attention.forward(ctx,attention_output);
        attention_output = layer_scale1.forward(ctx, attention_output);
        x = ggml_add(ctx, attention_output, x);
        ggml_tensor_t *layer_output = norm2.forward(ctx, x);

        layer_output = mlp.forward(ctx, layer_output);
        layer_output = layer_scale2.forward(ctx, layer_output);
        layer_output = ggml_add(ctx, layer_output, x);

    	return layer_output;
    }
};


struct Dinov2Encoder {
    struct Dinov2Layer layers[40];

    void create_weight_tensors(struct ggml_context* ctx) {
        for (int i = 0; i < 40; i++) {
            layers[i].create_weight_tensors(ctx);
        }
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        for (int i = 0; i < 40; i++) {
            snprintf(s, sizeof(s), "%slayer.%d.", prefix, i);
            layers[i].setup_weight_names(s);
        }
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        for (int i = 0; i < 40; i++) {
            x = layers[i].forward(ctx, x);
        }
    	return x;
    }
};

struct Dinov2PatchEmbeddings {
    const int num_channels = 3;
    const int hidden_size = 1536;

    struct Conv2d projection;

    void create_weight_tensors(struct ggml_context* ctx) {
        projection.in_channels = num_channels;
        projection.out_channels = hidden_size;

        // Fixed defaults ...
        projection.kernel_size = { 14, 14 };
        projection.stride = { 14, 14 };
        projection.padding = { 0, 0 };
        projection.dilation = { 1, 1 };
        projection.is_depthwise = false;
        projection.has_bias = true;

        projection.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "projection.");
        projection.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = projection.forward(ctx, x);

        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];
        x = ggml_reshape_3d(ctx, x, H*W, C, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

    	return x;
    }
};

struct Dinov2Embeddings {
    const int num_patches = 1369;
    const int hidden_size = 1536;
    // const int patch_size = 14;

    ggml_tensor_t* cls_token;
    ggml_tensor_t* mask_token;
    struct Dinov2PatchEmbeddings patch_embeddings;
    ggml_tensor_t* position_embeddings;

    void create_weight_tensors(struct ggml_context* ctx) {
        cls_token = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_size, 1, 1);
        mask_token = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 1);
        patch_embeddings.create_weight_tensors(ctx);
        position_embeddings = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_size, num_patches + 1, 1);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        ggml_format_name(cls_token, "%s%s", prefix, "cls_token");
        ggml_format_name(mask_token, "%s%s", prefix, "mask_token");

        snprintf(s, sizeof(s), "%s%s", prefix, "patch_embeddings.");
        patch_embeddings.setup_weight_names(s);

        ggml_format_name(position_embeddings, "%s%s", prefix, "position_embeddings");
    }


    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // x    f32 [518, 518, 3, 1], net.input_0
        int B = (int)x->ne[3];
        ggml_tensor_t *embeddings = patch_embeddings.forward(ctx, x);
        ggml_tensor_t *cls_tokens = ggml_repeat_ext(ctx, cls_token, 1, 1, 1, B);
        embeddings = ggml_concat(ctx, cls_tokens, embeddings, 1 /*dim*/);
        embeddings = ggml_add(ctx, embeddings, position_embeddings);
        // embeddings3    f32 [1536, 1370, 1, 1], 

    	return embeddings;
    }
};


struct Dinov2Network : ggml::GGMLNetwork {
    struct Dinov2Embeddings embeddings;
    struct Dinov2Encoder encoder;
    struct LayerNorm layernorm;

    void create_weight_tensors(struct ggml_context* ctx) {
        embeddings.create_weight_tensors(ctx);
        encoder.create_weight_tensors(ctx);

        layernorm.normalized_shape = 1536;
        layernorm.eps = 1e-6;
        layernorm.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "embeddings.");
        embeddings.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "encoder.");
        encoder.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "layernorm.");
        layernorm.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[])
    {
        GGML_ASSERT(argc == 1);
        ggml_tensor_t *x = argv[0];

        // x = ggml_nn_arange(ctx, x);
        x = embeddings.forward(ctx, x);
        x = encoder.forward(ctx, x);
        x = layernorm.forward(ctx, x);

        return x;
    }    
};


// struct Dinov2Model {
//     Dinov2Network dinov2_net;
//     ggml::GGMLModel model;

//     int init(int device)
//     {
//         // -----------------------------------------------------------------------------------------
//         dinov2_net.set_device(device);
//         dinov2_net.start_engine();
//         // dinov2_net.dump();

//         check_point(model.preload("models/image3d_shape.gguf") == RET_OK);

//         // load weights ...
//         dinov2_net.load_weight(&model, "shape_dinov2.");
//         model.clear();

//         return RET_OK;
//     }

//     TENSOR* forward(TENSOR* image)
//     {
//         TENSOR* argv[1];
//         argv[0] = image;

//         // tensor [image] size: [1, 3, 518, 518], min: -2.099609, max: 2.638672, mean: 1.449731
//         TENSOR* y = dinov2_net.engine_forward(ARRAY_SIZE(argv), argv);
//         // tensor [y] size: [1, 1370, 1536], min: -16.265625, max: 12.71875, mean: -0.01448

//         return y;
//     }

//     void exit()
//     {
//         dinov2_net.stop_engine();
//     }
// };

#endif // __DINOV2__H__
