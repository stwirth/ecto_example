#include <ecto/ecto.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

namespace ecto_example
{

using ecto::tendrils;

struct DescriptorMatcherKnn
{
  static void declare_params(tendrils& params)
  {
    params.declare<std::string>("matcher_type", "The descriptor matcher OpenCV type.", "BruteForce");
    params.declare<int>("k", "The number of neighbors to search for matching", 2);
  }

  static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare<cv::Mat>("train_descriptors", "Descriptors for training. These form the data base.");
    inputs.declare<cv::Mat>("test_descriptors", "Descriptors for testing. These will be used to query.");
    outputs.declare<std::vector<std::vector<cv::DMatch> > >("knn_matches", "The knn matches.");
  }

  void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
  {
    // this throws if create() fails
    descriptor_matcher_ = cv::DescriptorMatcher::create(params.get<std::string>("matcher_type"));
    k_ = params.get<int>("k");
  }
 
  int process(const tendrils& input, const tendrils& output)
  {
    cv::Mat train_descriptors, test_descriptors;
    input["train_descriptors"] >> train_descriptors;
    input["test_descriptors"] >> test_descriptors;

    std::vector<std::vector<cv::DMatch> > knn_matches;
    descriptor_matcher_->knnMatch(test_descriptors, train_descriptors,
            knn_matches, k_);

    output["knn_matches"] << knn_matches;

    return ecto::OK;
  }

  cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;
  int k_;

};

}

ECTO_CELL(ecto_example, ecto_example::DescriptorMatcherKnn, "DescriptorMatcherKnn", "Matches descriptors, looks for k neighbors.");

