/************************************************************************************
***
*** Copyright 2025 Dell Du(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Tue 10 Jun 2025 12:42:29 PM CST
***
************************************************************************************/

#include "dinov2.h"
#include "dit.h"
#include "shapevae.h"
#include "image.h"

struct ShapeModel {
private:
    bool model_valid_ = false;
    ggml::GGMLModel model_weight_;
   
public:
    const int MAX_H = 518;
    const int MAX_W = 518;
    const int MAX_TIMES = 1;

    Tensor4f dinov2_output;
    Tensor4f dit_output;
    Tensor4f vae_output;
    Tensor4f geo_output;

    ShapeModel()
    {
        model_valid_ = model_weight_.load("models/image3d_shape.gguf");
    }

	~ShapeModel() {
		// model_weight_.clear();
		model_valid_ = false;
	}

    inline bool valid() const { return model_valid_; }


    bool dinov2_forward(int device, Tensor4f* image)
    {
        Dinov2Network dinov2_net;
        ggml::Model dinov2_model;

        if (!dinov2_model.load(&dinov2_net, device, model_weight_, "shape_dinov2."))
            return false;

        std::vector<Tensor4f> x { *image };
        if (!dinov2_model.forward(x))
            return false;

        // tensor [image] size: [1, 3, 518, 518], min: -2.099609, max: 2.638672, mean: 1.449731
        dinov2_output = dinov2_model.get_output_tensor(0);
        // tensor [y] size: [1, 1370, 1536], min: -16.265625, max: 12.71875, mean: -0.01448

        return true;
    }


    bool dit_forward(int device, Tensor4f* x, Tensor4f* t, Tensor4f* cond)
    {
        DiTNetwork dit_net;
        ggml::Model dit_model;

        if (!dit_model.load(&dit_net, device, model_weight_, "shape_dit."))
            return false;

        std::vector<Tensor4f> xs { *x, *t, *cond };
        if (!dit_model.forward(xs))
            return false;

        // tensor [x] size: [2, 512, 64], min: -4.179688, max: 4.238281, mean: 0.005243
        // tensor [t] size: [2], min: 0.0, max: 0.0, mean: 0.0
        // tensor [cond] size: [2, 1370, 1536], min: -15.28125, max: 14.375, mean: -0.009306
        dit_output = dit_model.get_output_tensor(0);
        // tensor [y] size: [2, 512, 64], min: -4.257812, max: 4.207031, mean: -0.006032

        return true;
    }


    bool vae_forward(int device, Tensor4f* latents)
    {
        ShapeVaeNetwork vae_net;
        ggml::Model vae_model;

        if (!vae_model.load(&vae_net, device, model_weight_, "shape_vae."))
            return false;

        std::vector<Tensor4f> x { *latents };
        if (! vae_model.forward(x))
            return false;

        // # tensor [latents] size: [1, 512, 64], min: -4.003906, max: 3.90625, mean: 0.018309
        vae_output = vae_model.get_output_tensor(0);
        // # tensor [latents] size: [1, 512, 1024], min: -374.5, max: 37.09375, mean: 0.019848

        return true;
    }


    bool geo_forward(int device, Tensor4f* queries, Tensor4f* latents)
    {
        GeoDecoder geo_net;
        ggml::Model geo_model;

        if (! geo_model.load(&geo_net, device, model_weight_, "shape_vae.geo_decoder."))
            return false;

        std::vector<Tensor4f> x { *queries, *latents };
        if (! geo_model.forward(x))
            return false;

        // # tensor [queries] size: [1, 8000, 3], min: -1.009766, max: 1.009766, mean: -0.658704
        // # tensor [latents] size: [1, 512, 1024], min: -369.75, max: 36.4375, mean: 0.016268
        geo_output = geo_model.get_output_tensor(0);
        // # tensor [latents] size: [1, 512, 1024], min: -374.5, max: 37.09375, mean: 0.019848

        return true;
    }
};

int image3d_predict(int device, int argc, char *argv[], char* output_dir)
{
	char *p, output_filename[1024];

	ShapeModel shape_model;

	if (! shape_model.valid())
		return -1;

	for (int i = 0; i < argc; i++) {
	    redos::Image4f image;
	    if (!image.load(argv[i])) {
	    	continue;
	    }

        Tensor4f input_tensor = image.tensor(1 /*dim Channel */, 0, 3);
        input_tensor = redos::tensor_interpolate(input_tensor, 518, 518);
	    if (! shape_model.dinov2_forward(device, &input_tensor)) {
	    	continue;
	    }
	    redos::tensor_show("dinov2", shape_model.dinov2_output);
        // Tensor dinov2 [1, 1, 1370, 1536]

        // int C1 = shape_model.dinov2_output.dimension(1); // 1
        // int C2 = shape_model.dinov2_output.dimension(2); // 1370
        // int C3 = shape_model.dinov2_output.dimension(3); // 1536
        // Eigen::array<Eigen::Index, 3> dims = { C1, C2, C3};
        // Tensor3f dit_cond1 = shape_model.dinov2_output.reshape(dims);
        Tensor4f dit_zeros = shape_model.dinov2_output;
        dit_zeros.setConstant(0.0f);
        Tensor4f dit_cond = redos::tensor_concat(shape_model.dinov2_output, dit_zeros, 1/*dim*/);
        redos::tensor_show("dit_cond", dit_cond);

        Tensor4f latents(1, 1, 512, 64);
        redos::tensor_randn(latents);
        redos::tensor_show("latents", latents);

        const float guidance_scale = 5.0f;
        int num_inference_steps = 5;
        float step_scale = 1.0f/(num_inference_steps - 1.0f);
        // Tensor1f sigmas(num_inference_steps);
        // redos::tensor_arange(sigmas, 0, 1.0f/num_inference_steps);
        // redos::tensor_show("sigmas", sigmas);

        // for i in range(num_inference_steps):
        //     pbar.update(1)
        //     latent_model_input = torch.cat([latents] * 2, dim=0) # size() -- [2, 512, 64]


        //     timestep = sigmas[i].expand(2)
        //     noise_pred = self.shape_dit(latent_model_input, timestep, dit_condition) # xxxx_9999
        //     # tensor [noise_pred] size: [2, 512, 64], min: -3.826172, max: 3.9375, mean: 0.000697

        //     noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        //     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        //     if (i < num_inference_steps - 1):
        //         latents = latents + step_scale * noise_pred

        Tensor4f timestep(1, 1, 1, 2);

        for (int n = 0; n < num_inference_steps; n++) {
            printf("------------- progress %d ...\n", n);

            Tensor4f latent_model_input = redos::tensor_concat(latents, latents, 1/*dim*/);
            timestep.setConstant(n * 1.0f/num_inference_steps); // sigmas[n]

            if (! shape_model.dit_forward(device, &latent_model_input, &timestep, &dit_cond)) {
                continue;
            }

            Tensor4f noise_pred_cond = redos::tensor_slice(shape_model.dit_output, 1/*dim*/, 0/*start*/, 1/*stop*/);
            Tensor4f noise_pred_uncond = redos::tensor_slice(shape_model.dit_output, 1/*dim*/, 1/*start*/, 2/*stop*/);

            Tensor4f noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond);

            latents = latents + step_scale * noise_pred;
        }
        redos::tensor_show("latents2", latents);


        if (! shape_model.vae_forward(device, &latents)) {
            continue;
        }
        redos::tensor_show("====> vae_output", shape_model.vae_output);


        p = strrchr(argv[i], '/');
        if (p != NULL) {
            snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, p + 1);
        } else {
            snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, argv[i]);
        }
	}

    return 0;
}
