#ifndef DEPTH_ENHANCER_H
#define DEPTH_ENHANCER_H

#include <opencv2/opencv.hpp>
#include <deque>
#include <unordered_map>
#include <mutex>
#include <Eigen/Dense>
#include <memory>

namespace ORB_SLAM3 {

struct Point3D {
    Eigen::Vector3f position;
    std::vector<float> observations;
    std::vector<int> frameIds;
    std::vector<double> timestamps;
    int observationCount;
    float confidence;
    
    Point3D() : observationCount(0), confidence(0.0f) {}
    
    void addObservation(float depth, int frameId, double timestamp);
    float getWeightedDepth(double currentTime) const;
    void updateConfidence();
};

class DepthEnhancer {
public:
    struct DepthFrame {
        cv::Mat depth;
        cv::Mat rgb;
        Eigen::Matrix4f pose;
        double timestamp;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        int frameId;
        bool isKeyFrame;
    };
    
    struct Config {
        // Jetson Nano optimized defaults
        int maxHistorySize = 5;      // Reduced for memory
        int maxKeyFrames = 3;        // Reduced for memory
        float depthScale = 5000.0f;
        float maxDepth = 8.0f;       // Reduced range
        float minDepth = 0.1f;
        int patchSize = 3;           // Smaller patch
        float outlierThreshold = 0.3f;
        
        // Downsampling for performance
        int processingStep = 2;      // Process every 2nd pixel
        int pyramidLevels = 2;       // Reduced pyramid levels
        
        // Camera intrinsics
        float fx, fy, cx, cy;
    };

private:
    // Core data structures
    std::deque<DepthFrame> mDepthHistory;
    std::deque<DepthFrame> mKeyFrameHistory;
    std::unordered_map<uint64_t, std::shared_ptr<Point3D>> m3DPointPool;
    
    // Configuration
    Config mConfig;
    
    // Frame counter
    int mFrameCounter;
    
    // Thread safety
    std::mutex mMutex;
    
    // Performance monitoring
    struct TimingStats {
        double projection = 0;
        double correspondence = 0;
        double fusion = 0;
        double refinement = 0;
        double total = 0;
    } mLastTiming;
    
    // Internal matrices
    cv::Mat mFeatureConfidenceMap;
    cv::Mat mDepthValidityMask;
    cv::Mat mK;

public:
    DepthEnhancer(const Config& config);
    ~DepthEnhancer();
    
    // Main enhancement function
    cv::Mat enhanceDepth(const cv::Mat& rawDepth, 
                        const cv::Mat& rgb,
                        const Eigen::Matrix4f& currentPose,
                        const std::vector<cv::KeyPoint>& orbFeatures,
                        const cv::Mat& orbDescriptors,
                        double timestamp,
                        bool isKeyFrame = false);
    
    // Get timing statistics
    TimingStats getLastTiming() const { return mLastTiming; }
    
    // Clear history
    void reset();

private:
    // Core processing functions
    void temporalFusion(cv::Mat& depth, const Eigen::Matrix4f& currentPose, double timestamp);
    void spatialRefinement(cv::Mat& depth, const std::vector<cv::KeyPoint>& features);
    void updateHistory(const DepthFrame& frame);
    void update3DPointPool(const DepthFrame& frame);
    
    // Helper functions
    cv::Mat projectDepthToFrame(const cv::Mat& sourceDepth, 
                               const Eigen::Matrix4f& sourcePose,
                               const Eigen::Matrix4f& targetPose);
    
    void findCorrespondences(const cv::Mat& currentDepth,
                           const Eigen::Matrix4f& currentPose,
                           const DepthFrame& histFrame,
                           cv::Mat& correspondenceMap,
                           cv::Mat& confidenceMap);
    
    float computeNCC(const cv::Mat& patch1, const cv::Mat& patch2);
    float computeGeometricConsistency(int x, int y, float currentD, float projD,
                                     const cv::Mat& currentDepth, 
                                     const cv::Mat& projectedDepth);
    
    Eigen::Vector3f pixel2Camera(int x, int y, float depth);
    cv::Point2f camera2Pixel(const Eigen::Vector3f& p3d);
    
    uint64_t pointToKey(int x, int y, float depth);
    
    void removeOutliers(cv::Mat& depthPatch, float centerDepth);
    
    void updateFeatureConfidenceMap(const std::vector<cv::KeyPoint>& features, 
                                   const cv::Mat& depth);
    
    bool isValidDepth(float depth) const {
        return depth > mConfig.minDepth && depth < mConfig.maxDepth;
    }
    
    // Downsampling helper
    void processDownsampled(const cv::Mat& input, cv::Mat& output, 
                           std::function<void(int, int)> processFunc);
};

} // namespace ORB_SLAM3

#endif // DEPTH_ENHANCER_H