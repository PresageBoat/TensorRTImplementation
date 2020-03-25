
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include "opencv2/opencv.hpp"
using namespace cv;

static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int CHANNELS = 3;
static const int OUTPUT_SIZE = 1000;
static Logger gLogger;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";


void CaffeToGIEModel(const std::string& deployfile,//name for caffe prototxt
    const std::string& modelfile,//name for mode
    const std::vector<std::string>& outputs,//network outputs
    unsigned int maxBatchSize,//batch size -NB must be at least  as large as the batch we want to run with
    IHostMemory* &gieModelStream)//output buffer for the GIE model
{
    //create the builder
    IBuilder* builder = createInferBuilder(gLogger);
    //parse the caffe model to populate the network,then set the output
    INetworkDefinition* network = builder->createNetwork();
    nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
    const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor = parser->parse(
        deployfile.c_str(),
        modelfile.c_str(),
        *network,
        nvinfer1::DataType::kHALF);
    //specify which tensors are outputs
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    //build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);
    //
    network->destroy();
    parser->destroy();
    //we don't need the network any more,and we can destroy the parser
    gieModelStream = engine->serialize();
    engine->destroy();
    builder->destroy();
    nvcaffeparser1::shutdownProtobufLibrary();
}

void DoInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchsize) {
    const nvinfer1::ICudaEngine& engine = context.getEngine();
    //input and output buffer pointers that we pass to the engine -the engine requires exactly IEngine::getNbBindings()
    //of these ,but in this case we know that there is exactly one input and one output
    assert(engine.getNbBindings() == 2);
    void* buffers[2];
    //in order to bind the buffers, we need to know the names of the input and output tensors,
    //note tha indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
        outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    //create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchsize*CHANNELS*INPUT_H*INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchsize*OUTPUT_SIZE * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    //DMA the input to the GPU,execute the batch  asynchrously ,and DMA it back
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchsize*CHANNELS*INPUT_H*INPUT_W * sizeof(float),
        cudaMemcpyHostToDevice, stream));
    context.enqueue(batchsize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchsize*OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    //release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main()
{
    //caffe model 2 GIE
    //create a GIE model from the caffe model and serialize it to a stream
    IHostMemory* gitModelStream(nullptr);
    CaffeToGIEModel("path/resnet/ResNet-50-deploy.prototxt", 
        "path/resnet/ResNet-50-model.caffemodel",
        std::vector<std::string>{OUTPUT_BLOB_NAME},
        1, 
        gitModelStream);

    //de-serialize the engine

    //反序列化前向引擎
    nvinfer1::IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(gitModelStream->data(),
        gitModelStream->size(), nullptr);
    if (gitModelStream)
    {
        gitModelStream->destroy();
    }
    //前向推断
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();


    //const float mean_data[] = { 103.94 ,116.78 ,123.68 };
    const float mean_data[] = { 104 ,117 ,123 };
    const float scale = 0.017;


    //
    const string folder_path = "F:/imageNet/val/";
    const string val_folder_path = "F:/imageNet/";
    ifstream ifs(val_folder_path + "val.txt");
    string line_str;
    vector<pair<string, int>> lines_;
    while (getline(ifs, line_str))
    {
        istringstream iss(line_str);
        string imgfn;
        iss >> imgfn;
        int segfn = 0;
        iss >> segfn;
        lines_.emplace_back(std::make_pair(imgfn, segfn));
    }
    int total_true = 0;

    double  total_time = 0.f;
    std::cout << "begin test inference\n" << std::endl;

    for (int i = 0; i < lines_.size(); i++)
    //for (int i = 0; i < 10000; i++)
    {
        Mat src = imread(folder_path + lines_[i].first);
        //短边缩放256 ，然后中心crop224
        int img_width = src.cols;
        int img_height = src.rows;
        int size = 256;
        if (img_width <= img_height)
        {
            int new_size = cvRound((1.0*img_height / img_width) *size);
            resize(src, src, cv::Size(size, new_size));
        }
        else
        {
            int new_size = cvRound((1.0*img_width / img_height)  *size);
            resize(src, src, cv::Size(new_size, size));
        }
        Mat dst_input;//输入的用于分类的图像，大小为224*224
        int w_off = 0, h_off = 0;
        w_off = (int)(src.cols - 224) / 2;
        h_off = (int)(src.rows - 224) / 2;
        cv::Rect crop(w_off, h_off, 224, 224);
        src(crop).copyTo(dst_input);

        //
        float data[INPUT_H*INPUT_W*CHANNELS];
        float *pdata = data;
        for (int c = 0; c < CHANNELS; ++c)
        {
            for (int h = 0; h < INPUT_H; ++h)
            {
                for (int w = 0; w < INPUT_W; ++w)
                {
                    *pdata++ = float((float)(dst_input.at<Vec3b>(h, w)[c]) - (float)(mean_data[c]));
                }
            }
        }



        //run inference
        float prob[OUTPUT_SIZE];
        double start, end;
        start = (double)cvGetTickCount();
        DoInference(*context, data, prob, 1);
        end = (double)cvGetTickCount();
        total_time = end - start;

        //取最大值，判断类别
        int max_index = 0;
        float max_prob = 0.f;
        for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
        {
            if (prob[i] > max_prob)
            {
                max_prob = prob[i];
                max_index = i;
            }
        }
        if (max_index == lines_[i].second)
        {
            total_true++;
        }

        if (i % 1000 == 0)
        {
            cout << total_true << "/" << i << endl;
        }
    }

    //cout << "平均分类时间：" << total_time / 10000 << " ms" << endl;
    //cout << "平均准确率：" << total_true *1.0 / 10000 << endl;
    cout << "平均分类时间：" << total_time / lines_.size() << " ms" << endl;
    cout << "平均准确率：" << total_true *1.0 / lines_.size() << endl;


    //destory the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    getchar();
}
