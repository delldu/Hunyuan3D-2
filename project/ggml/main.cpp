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
// #include <string.h>
// #include <unistd.h>

#define DEFAULT_DEVICE 1
#define DEFAULT_OUTPUT "output"

int image3d_predict(int device, int argc, char *argv[], char* output_dir);

static void image3d_help(char* cmd)
{
    printf("Usage: %s [option] gray_files\n", cmd);
    printf("    -h, --help                   Display this help.\n");
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

    return image3d_predict(device_no, argc - optind, &argv[optind], output_dir);
}
