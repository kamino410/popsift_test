#include <stdlib.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <popsift/common/device_prop.h>
#include <popsift/features.h>
#include <popsift/popsift.h>
#include <popsift/sift_conf.h>

#include <opencv2/opencv.hpp>

using namespace std;

static bool print_dev_info = false;
static bool print_time_info = false;
static bool write_as_uchar = false;
static bool dont_write = false;
static bool pgmread_loading = false;
static bool float_mode = false;

static void parseargs(int argc, char** argv, popsift::Config& config, string& inputFile) {
  using namespace boost::program_options;

  options_description options("Options");
  {
    options.add_options()("help,h", "Print usage")(
        "verbose,v", bool_switch()->notifier([&](bool i) {
          if (i) config.setVerbose();
        }),
        "")("log,l", bool_switch()->notifier([&](bool i) {
      if (i) config.setLogMode(popsift::Config::All);
    }),
            "Write debugging files")

        ("input-file,i", value<std::string>(&inputFile)->required(), "Input file");
  }
  options_description parameters("Parameters");
  {
    parameters.add_options()("octaves", value<int>(&config.octaves), "Number of octaves")(
        "levels", value<int>(&config.levels), "Number of levels per octave")(
        "sigma", value<float>()->notifier([&](float f) { config.setSigma(f); }),
        "Initial sigma value")

        ("threshold", value<float>()->notifier([&](float f) { config.setThreshold(f); }),
         "Contrast threshold")("edge-threshold",
                               value<float>()->notifier([&](float f) { config.setEdgeLimit(f); }),
                               "On-edge threshold")(
            "edge-limit", value<float>()->notifier([&](float f) { config.setEdgeLimit(f); }),
            "On-edge threshold")(
            "downsampling", value<float>()->notifier([&](float f) { config.setDownsampling(f); }),
            "Downscale width and height of input by 2^N")(
            "initial-blur", value<float>()->notifier([&](float f) { config.setInitialBlur(f); }),
            "Assume initial blur, subtract when blurring first time");
  }
  options_description modes("Modes");
  {
    modes.add_options()("gauss-mode", value<std::string>()->notifier([&](const std::string& s) {
      config.setGaussMode(s);
    }),
                        popsift::Config::getGaussModeUsage())
        // "Choice of span (1-sided) for Gauss filters. Default is VLFeat-like computation depending
        // on sigma. " "Options are: vlfeat, relative, relative-all, opencv, fixed9, fixed15"
        ("desc-mode",
         value<std::string>()->notifier([&](const std::string& s) { config.setDescMode(s); }),
         "Choice of descriptor extraction modes:\n"
         "loop, iloop, grid, igrid, notile\n"
         "Default is loop\n"
         "loop is OpenCV-like horizontal scanning, computing only valid points, grid extracts only "
         "useful points but rounds them, iloop uses linear texture and rotated gradiant fetching. "
         "igrid is grid with linear interpolation. notile is like igrid but avoids redundant "
         "gradiant fetching.")(
            "popsift-mode", bool_switch()->notifier([&](bool b) {
              if (b) config.setMode(popsift::Config::PopSift);
            }),
            "During the initial upscale, shift pixels by 1. In extrema refinement, steps up to "
            "0.6, do not reject points when reaching max iterations, "
            "first contrast threshold is .8 * peak thresh. Shift feature coords octave 0 back to "
            "original pos.")(
            "vlfeat-mode", bool_switch()->notifier([&](bool b) {
              if (b) config.setMode(popsift::Config::VLFeat);
            }),
            "During the initial upscale, shift pixels by 1. That creates a sharper upscaled image. "
            "In extrema refinement, steps up to 0.6, levels remain unchanged, "
            "do not reject points when reaching max iterations, "
            "first contrast threshold is .8 * peak thresh.")(
            "opencv-mode", bool_switch()->notifier([&](bool b) {
              if (b) config.setMode(popsift::Config::OpenCV);
            }),
            "During the initial upscale, shift pixels by 0.5. "
            "In extrema refinement, steps up to 0.5, "
            "reject points when reaching max iterations, "
            "first contrast threshold is floor(.5 * peak thresh). "
            "Computed filter width are lower than VLFeat/PopSift")(
            "direct-scaling", bool_switch()->notifier([&](bool b) {
              if (b) config.setScalingMode(popsift::Config::ScaleDirect);
            }),
            "Direct each octave from upscaled orig instead of blurred level.")(
            "norm-multi",
            value<int>()->notifier([&](int i) { config.setNormalizationMultiplier(i); }),
            "Multiply the descriptor by pow(2,<int>).")(
            "norm-mode",
            value<std::string>()->notifier([&](const std::string& s) { config.setNormMode(s); }),
            popsift::Config::getNormModeUsage())("root-sift", bool_switch()->notifier([&](bool b) {
          if (b) config.setNormMode(popsift::Config::RootSift);
        }),
                                                 popsift::Config::getNormModeUsage())(
            "filter-max-extrema",
            value<int>()->notifier([&](int f) { config.setFilterMaxExtrema(f); }),
            "Approximate max number of extrema.")(
            "filter-grid", value<int>()->notifier([&](int f) { config.setFilterGridSize(f); }),
            "Grid edge length for extrema filtering (ie. value 4 leads to a 4x4 grid)")(
            "filter-sort", value<std::string>()->notifier([&](const std::string& s) {
              config.setFilterSorting(s);
            }),
            "Sort extrema in each cell by scale, either random (default), up or down");
  }
  options_description informational("Informational");
  {
    informational.add_options()("print-gauss-tables", bool_switch()->notifier([&](bool b) {
      if (b) config.setPrintGaussTables();
    }),
                                "A debug output printing Gauss filter size and tables")(
        "print-dev-info", bool_switch(&print_dev_info)->default_value(false),
        "A debug output printing CUDA device information")(
        "print-time-info", bool_switch(&print_time_info)->default_value(false),
        "A debug output printing image processing time after load()")(
        "write-as-uchar", bool_switch(&write_as_uchar)->default_value(false),
        "Output descriptors rounded to int.\n"
        "Scaling to sensible ranges is not automatic, should be combined with --norm-multi=9 or "
        "similar")("dont-write", bool_switch(&dont_write)->default_value(false),
                   "Suppress descriptor output")(
        "pgmread-loading", bool_switch(&pgmread_loading)->default_value(false),
        "Use the old image loader instead of LibDevIL")(
        "float-mode", bool_switch(&float_mode)->default_value(false),
        "Upload image to GPU as float instead of byte");

    //("test-direct-scaling")
  }

  options_description all("Allowed options");
  all.add(options).add(parameters).add(modes).add(informational);
  variables_map vm;

  try {
    store(parse_command_line(argc, argv, all), vm);

    if (vm.count("help")) {
      std::cout << all << '\n';
      exit(1);
    }

    notify(vm);  // Notify does processing (e.g., raise exceptions if required args are missing)
  } catch (boost::program_options::error& e) {
    std::cerr << "Error: " << e.what() << std::endl << std::endl;
    std::cerr << "Usage:\n\n" << all << std::endl;
    exit(EXIT_FAILURE);
  }
}

static void collectFilenames(list<string>& inputFiles, const boost::filesystem::path& inputFile) {
  vector<boost::filesystem::path> vec;
  std::copy(boost::filesystem::directory_iterator(inputFile),
            boost::filesystem::directory_iterator(), std::back_inserter(vec));
  for (const auto& currPath : vec) {
    if (boost::filesystem::is_regular_file(currPath)) {
      inputFiles.push_back(currPath.string());
    } else if (boost::filesystem::is_directory(currPath)) {
      collectFilenames(inputFiles, currPath);
    }
  }
}

SiftJob* process_image(const cv::Mat& img, PopSift& PopSift) {
  int w;
  int h;
  SiftJob* job;
  unsigned char* image_data;

  {
    image_data = img.data;
    w = img.cols;
    h = img.rows;
    if (image_data == 0) { exit(-1); }

    if (not float_mode) {
      // PopSift.init( w, h );
      job = PopSift.enqueue(w, h, image_data);

      // delete[] image_data;
    } else {
      float* f_image_data = new float[w * h];
      for (int i = 0; i < w * h; i++) { f_image_data[i] = float(image_data[i]) / 256.0f; }
      job = PopSift.enqueue(w, h, f_image_data);

      // delete[] image_data;
      delete[] f_image_data;
    }
  }

  return job;
}

void read_job(SiftJob* job, cv::Mat img, bool really_write) {
  popsift::Features* feature_list = job->get();
  cerr << "Feature points: " << feature_list->getFeatureCount()
       << ", Descriptors: " << feature_list->getDescriptorCount() << endl;

  // if (feature_list->getFeatureCount() > 0) { cerr << feature_list->getFeatures()[0].xpos << endl;
  // }
  std::vector<cv::KeyPoint> keypoints;
  for (int i = 0; i < feature_list->getFeatureCount(); i++) {
    auto fp = feature_list->getFeatures()[i];
    auto kp = cv::KeyPoint(fp.xpos, fp.ypos, fp.sigma * 100);
    keypoints.push_back(kp);
  }
  cv::Mat viz;
  cv::drawKeypoints(img, keypoints, viz);
  cv::imwrite("test.png", viz);

  if (really_write) {
    std::ofstream of("output-features.txt");
    feature_list->print(of, write_as_uchar);
  }
  delete feature_list;
}

int main(int argc, char** argv) {
  cudaDeviceReset();

  popsift::Config config;
  list<string> inputFiles;
  string inputFile = "";
  const char* appName = argv[0];

  try {
    parseargs(argc, argv, config, inputFile);  // Parse command line
    std::cout << inputFile << std::endl;
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
    exit(1);
  }

  if (boost::filesystem::exists(inputFile)) {
    if (boost::filesystem::is_directory(inputFile)) {
      cout << "BOOST " << inputFile << " is directory" << endl;
      collectFilenames(inputFiles, inputFile);
      if (inputFiles.empty()) {
        cerr << "No files in directory, nothing to do" << endl;
        exit(0);
      }
    } else if (boost::filesystem::is_regular_file(inputFile)) {
      inputFiles.push_back(inputFile);
    } else {
      cout << "Input file is neither regular file nor directory, nothing to do" << endl;
      exit(-1);
    }
  }

  popsift::cuda::device_prop_t deviceInfo;
  deviceInfo.set(0, print_dev_info);
  if (print_dev_info) deviceInfo.print();

  PopSift PopSift(config, popsift::Config::ExtractingMode,
                  float_mode ? PopSift::FloatImages : PopSift::ByteImages);

  // std::queue<SiftJob*> jobs;
  // for (auto it = inputFiles.begin(); it != inputFiles.end(); it++) {
  //   inputFile = it->c_str();
  //   cv::Mat img = cv::imread(inputFile, 0);

  //   SiftJob* job = process_image(img, PopSift);
  //   jobs.push(job);
  // }

  cv::VideoCapture cap;
  cap.open(0);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 2560);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1440);
  cap.set(cv::CAP_PROP_FRAME_COUNT, 32);

  if (!cap.isOpened()) {
    std::cerr << "Unable to open camera" << std::endl;
    exit(-1);
  }

  {
    cv::Mat img;
    cap.read(img);
    std::cout << "Image size : " << img.rows << " x " << img.cols << std::endl;
  }

  std::queue<SiftJob*> jobs;
  std::queue<cv::Mat> images;
  for (int i = 0; i < 50; i++) {
    cv::Mat frame, gray;
    cap.read(frame);

    cv::cvtColor(frame, gray, CV_RGB2GRAY);

    SiftJob* job = process_image(gray, PopSift);
    jobs.push(job);
    images.push(frame);

    while (jobs.size() > 1) {
      SiftJob* job = jobs.front();
      jobs.pop();
      if (job) {
        read_job(job, images.front(), not dont_write);
        delete job;
      }
      images.pop();
    }
  }

  PopSift.uninit();
}

