/************************************************************************************
***
*** Copyright 2024 Dell Du(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Tue 02 Apr 2024 03:49:53 PM CST
***
************************************************************************************/

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
// #include <glob.h>

#include <ggml_engine.h>
#include <nimage/tensor.h>

#include "dinov2.h"
#include "dit.h"
#include "shapevae.h"

#define DEFAULT_DEVICE 1
#define DEFAULT_OUTPUT "output"

// int image3d_predict(Dinov2Model* dinov2_model, char* gray_files, char* output_dir);

static void image3d_help(char* cmd)
{
    printf("Usage: %s [option] gray_files\n", cmd);
    printf("    -h, --help                   Display this help, engine version %s.\n", GGML_ENGINE_VERSION);
    printf("    -d, --device <no>            Set device (0 -- cpu, 1 -- cuda0, 2 -- cuda1, ..., default: %d)\n", DEFAULT_DEVICE);
    printf("    -o, --output                 output dir, default: %s.\n", DEFAULT_OUTPUT);

    exit(1);
}

int main(int argc, char** argv)
{
    int optc;
    int option_index = 0;
    int device_no = DEFAULT_DEVICE;
    char* output_dir = (char*)DEFAULT_OUTPUT;

    struct option long_opts[] = { { "help", 0, 0, 'h' }, { "device", 1, 0, 'd' }, { "examples", 1, 0, 'e' },
        { "output", 1, 0, 'o' }, { 0, 0, 0, 0 }

    };

    printf("checkpoint 1 ...\n");

    if (argc <= 1)
        image3d_help(argv[0]);

    while ((optc = getopt_long(argc, argv, "h d: o:", long_opts, &option_index)) != EOF) {
        switch (optc) {
        case 'd':
            device_no = atoi(optarg);
            break;
        case 'o':
            output_dir = optarg;
            break;
        case 'h': // help
        default:
            image3d_help(argv[0]);
            break;
        }
    }

    printf("checkpoint 2 ...\n");

    // client
    if (optind == argc) // no input image, nothing to do ...
        return 0;

    Dinov2Model dinov2_model;

    // int network
    {
        dinov2_model.init(device_no);
    }

    printf("checkpoint 3 ... optind = %d, argc = %d\n", optind, argc);

    for (int i = optind; i < argc; i++) {
        // image3d_predict(&dinov2_model, argv[i], output_dir);
        printf("create from %s to %s ...\n", argv[i], output_dir);
    }
    TENSOR *x = tensor_create(1, 3, 518, 518);
    TENSOR *y = dinov2_model.forward(x);

    tensor_show("y", y);

    // Info: y Tensor: 1x1x1370x1536
    // min: -16.2236, max: 12.5019, mean: -0.0144
    // 0.9444 1.1059 1.5564 -1.8198 -0.1310 -0.8696 1.1567 -0.6701 -0.6838 2.2458 ... 0.3093 -0.5996 0.8670 -0.3768 0.2183 -0.1071 0.3080 -0.1961 1.3123 -0.0342 

    // # tensor [y] size: [1, 1370, 1536], min: -16.265625, max: 12.71875, mean: -0.01448

    tensor_destroy(y);
    tensor_destroy(x);


    // free network ...
    {
        dinov2_model.exit();
    }

    return 0;
}
