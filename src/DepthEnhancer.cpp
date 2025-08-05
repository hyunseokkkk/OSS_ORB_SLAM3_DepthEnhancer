#include "DepthEnhancer.h"
#include <chrono>
#include <algorithm>

using namespace std;
using namespace cv;

namespace ORB_SLAM3 {

// Point3D implementation
void Point3D::addObservation(float depth, int frameId, double timestamp) {
    observations.push_back(depth);
    frameIds.push_back(frameId);
    timestamps.push_back(timestamp);
    observationCount++;
    updateConfidence();
}

void Point3D::updateConfidence() {
    if (observationCount > 5) 
        confidence = 1.0f;
    else 
        confidence = observationCount / 5.0f;
}

float Point3D::getWeightedDepth(double currentTime) const {
    if (observations.empty()) return -1.0f;
    
    float weightedSum = 0.0f;
    float totalWeight = 0.0f;
    
    for (size_t i = 0; i < observations.size(); ++i) {
        double timeDiff = currentTime - timestamps[i];
        float timeWeight = exp(-timeDiff / 1.0);
        weightedSum += observations[i] * timeWeight;
        totalWeight += timeWeight;
    }
    
    return totalWeight > 0 ? weightedSum / totalWeight : -1.0f;
}

// DepthEnhancer implementation
DepthEnhancer::DepthEnhancer(const Config& config) 
    : mConfig(config), mFrameCounter(0) {
    
    mK = (Mat_<float>(3,3) << config.fx, 0, config.cx,
                              0, config.fy, config.cy,
                              0, 0, 1);
}

DepthEnhancer::~DepthEnhancer() {
    reset();
}

cv::Mat DepthEnhancer::enhanceDepth(const cv::Mat& rawDepth, 
                                    const cv::Mat& rgb,
                                    const Eigen::Matrix4f& currentPose,
                                    const std::vector<cv::KeyPoint>& orbFeatures,
                                    const cv::Mat& orbDescriptors,
                                    double timestamp,
                                    bool isKeyFrame) {
    auto startTime = chrono::high_resolution_clock::now();
    
    // Convert depth to float if needed
    Mat depth;
    if (rawDepth.type() == CV_16U) {
        rawDepth.convertTo(depth, CV_32F, 1.0f / mConfig.depthScale);
    } else {
        depth = rawDepth.clone();
    }
    
    // Create validity mask
    mDepthValidityMask = (depth > mConfig.minDepth) & (depth < mConfig.maxDepth);
    
    // Update feature confidence map
    updateFeatureConfidenceMap(orbFeatures, depth);
    
    // Temporal fusion if we have history
    auto fusionStart = chrono::high_resolution_clock::now();
    if (!mDepthHistory.empty()) {
        temporalFusion(depth, currentPose, timestamp);
    }
    mLastTiming.fusion = chrono::duration<double, milli>(
        chrono::high_resolution_clock::now() - fusionStart).count();
    
    // Spatial refinement around features
    auto refineStart = chrono::high_resolution_clock::now();
    spatialRefinement(depth, orbFeatures);
    mLastTiming.refinement = chrono::duration<double, milli>(
        chrono::high_resolution_clock::now() - refineStart).count();
    
    // Create current frame
    DepthFrame currentFrame;
    currentFrame.depth = depth.clone();
    currentFrame.rgb = rgb.clone();
    currentFrame.pose = currentPose;
    currentFrame.timestamp = timestamp;
    currentFrame.keypoints = orbFeatures;
    currentFrame.descriptors = orbDescriptors.clone();
    currentFrame.frameId = mFrameCounter++;
    currentFrame.isKeyFrame = isKeyFrame;
    
    // Update history and 3D point pool
    updateHistory(currentFrame);
    
    // Update 3D points only for keyframes to save computation
    if (isKeyFrame) {
        update3DPointPool(currentFrame);
    }
    
    mLastTiming.total = chrono::duration<double, milli>(
        chrono::high_resolution_clock::now() - startTime).count();
    
    return depth;
}

void DepthEnhancer::temporalFusion(cv::Mat& depth, 
                                  const Eigen::Matrix4f& currentPose, 
                                  double timestamp) {
    Mat fusedDepth = Mat::zeros(depth.size(), CV_32F);
    Mat totalWeight = Mat::zeros(depth.size(), CV_32F);
    
    // Process only recent frames
    int processedFrames = 0;
    for (const auto& histFrame : mDepthHistory) {
        // Skip if too old
        if (timestamp - histFrame.timestamp > 2.0) continue;
        if (++processedFrames > 3) break; // Limit to 3 frames for performance
        
        // Project historical depth to current frame
        Mat projectedDepth = projectDepthToFrame(histFrame.depth, 
                                               histFrame.pose, 
                                               currentPose);
        
        // Find correspondences with downsampling
        Mat correspondenceMap, confidenceMap;
        findCorrespondences(depth, currentPose, histFrame, 
                          correspondenceMap, confidenceMap);
        
        // Time-based weight
        float timeWeight = exp(-(timestamp - histFrame.timestamp) / 1.0f);
        
        // Accumulate with weights (downsampled)
        const int step = mConfig.processingStep;
        for (int y = 0; y < depth.rows; y += step) {
            for (int x = 0; x < depth.cols; x += step) {
                if (correspondenceMap.at<uchar>(y, x) > 0) {
                    float conf = confidenceMap.at<float>(y, x);
                    float w = timeWeight * conf;
                    
                    // Apply to block
                    for (int dy = 0; dy < step && y+dy < depth.rows; ++dy) {
                        for (int dx = 0; dx < step && x+dx < depth.cols; ++dx) {
                            fusedDepth.at<float>(y+dy, x+dx) += 
                                projectedDepth.at<float>(y+dy, x+dx) * w;
                            totalWeight.at<float>(y+dy, x+dx) += w;
                        }
                    }
                }
            }
        }
    }
    
    // Normalize and blend with current depth
    for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
            if (totalWeight.at<float>(y, x) > 0) {
                float fusedValue = fusedDepth.at<float>(y, x) / totalWeight.at<float>(y, x);
                float currentValue = depth.at<float>(y, x);
                
                // Use feature confidence to blend
                float confidence = mFeatureConfidenceMap.at<float>(y, x);
                
                // Check for outliers
                if (abs(fusedValue - currentValue) < mConfig.outlierThreshold) {
                    depth.at<float>(y, x) = (1 - confidence) * fusedValue + 
                                           confidence * currentValue;
                }
            }
        }
    }
}

cv::Mat DepthEnhancer::projectDepthToFrame(const cv::Mat& sourceDepth,
                                          const Eigen::Matrix4f& sourcePose,
                                          const Eigen::Matrix4f& targetPose) {
    Mat projectedDepth = Mat::zeros(sourceDepth.size(), CV_32F);
    
    // Compute relative transformation
    Eigen::Matrix4f relPose = targetPose.inverse() * sourcePose;
    Eigen::Matrix3f R = relPose.block<3,3>(0,0);
    Eigen::Vector3f t = relPose.block<3,1>(0,3);
    
    // Process with downsampling for speed
    const int step = mConfig.processingStep;
    for (int y = 0; y < sourceDepth.rows; y += step) {
        for (int x = 0; x < sourceDepth.cols; x += step) {
            float d = sourceDepth.at<float>(y, x);
            if (!isValidDepth(d)) continue;
            
            // Convert to 3D
            Eigen::Vector3f p3d = pixel2Camera(x, y, d);
            
            // Transform to target frame
            Eigen::Vector3f p3d_target = R * p3d + t;
            
            if (p3d_target.z() <= 0) continue;
            
            // Project to target image
            Point2f p2d = camera2Pixel(p3d_target);
            
            int px = round(p2d.x);
            int py = round(p2d.y);
            
            // Fill block
            if (px >= 0 && px < projectedDepth.cols && 
                py >= 0 && py < projectedDepth.rows) {
                for (int dy = 0; dy < step && py+dy < projectedDepth.rows; ++dy) {
                    for (int dx = 0; dx < step && px+dx < projectedDepth.cols; ++dx) {
                        projectedDepth.at<float>(py+dy, px+dx) = p3d_target.z();
                    }
                }
            }
        }
    }
    
    return projectedDepth;
}

void DepthEnhancer::findCorrespondences(const cv::Mat& currentDepth,
                                       const Eigen::Matrix4f& currentPose,
                                       const DepthFrame& histFrame,
                                       cv::Mat& correspondenceMap,
                                       cv::Mat& confidenceMap) {
    correspondenceMap = Mat::zeros(currentDepth.size(), CV_8U);
    confidenceMap = Mat::zeros(currentDepth.size(), CV_32F);
    
    int halfPatch = mConfig.patchSize / 2;
    const int step = mConfig.processingStep;
    
    // Convert to grayscale for matching
    Mat currentGray, histGray;
    cvtColor(histFrame.rgb, histGray, COLOR_BGR2GRAY);
    
    // Process with downsampling
    for (int y = halfPatch; y < currentDepth.rows - halfPatch; y += step) {
        for (int x = halfPatch; x < currentDepth.cols - halfPatch; x += step) {
            float d = currentDepth.at<float>(y, x);
            if (!isValidDepth(d)) continue;
            
            // Check 3D point pool
            uint64_t key = pointToKey(x, y, d);
            
            bool found = false;
            float confidence = 0.0f;
            
            {
                lock_guard<mutex> lock(mMutex);
                auto it = m3DPointPool.find(key);
                if (it != m3DPointPool.end()) {
                    found = true;
                    confidence = it->second->confidence;
                }
            }
            
            if (found) {
                // Fill block
                for (int dy = 0; dy < step && y+dy < currentDepth.rows; ++dy) {
                    for (int dx = 0; dx < step && x+dx < currentDepth.cols; ++dx) {
                        correspondenceMap.at<uchar>(y+dy, x+dx) = 255;
                        confidenceMap.at<float>(y+dy, x+dx) = confidence;
                    }
                }
            }
        }
    }
}

void DepthEnhancer::spatialRefinement(cv::Mat& depth, 
                                     const std::vector<cv::KeyPoint>& features) {
    // Process only regions around features
    for (const auto& kp : features) {
        int x = kp.pt.x;
        int y = kp.pt.y;
        int radius = max(10, (int)(kp.size * 1.2));
        
        // Extract patch
        Rect roi(x - radius, y - radius, 2*radius + 1, 2*radius + 1);
        if (roi.x < 0 || roi.y < 0 || 
            roi.x + roi.width > depth.cols || 
            roi.y + roi.height > depth.rows) continue;
        
        Mat patch = depth(roi).clone();
        float centerDepth = depth.at<float>(y, x);
        
        if (!isValidDepth(centerDepth)) continue;
        
        // Remove outliers
        removeOutliers(patch, centerDepth);
        
        // Apply simple median filter instead of bilateral (faster)
        Mat smoothed;
        medianBlur(patch, smoothed, 3);
        
        // Copy back
        smoothed.copyTo(depth(roi));
    }
}

void DepthEnhancer::removeOutliers(cv::Mat& depthPatch, float centerDepth) {
    vector<float> validDepths;
    
    // Collect valid depths
    for (int y = 0; y < depthPatch.rows; ++y) {
        for (int x = 0; x < depthPatch.cols; ++x) {
            float d = depthPatch.at<float>(y, x);
            if (isValidDepth(d) && abs(d - centerDepth) < mConfig.outlierThreshold) {
                validDepths.push_back(d);
            }
        }
    }
    
    if (validDepths.size() < 3) return;
    
    // Calculate median
    nth_element(validDepths.begin(), 
                validDepths.begin() + validDepths.size()/2, 
                validDepths.end());
    float median = validDepths[validDepths.size()/2];
    
    // Remove outliers
    for (int y = 0; y < depthPatch.rows; ++y) {
        for (int x = 0; x < depthPatch.cols; ++x) {
            float d = depthPatch.at<float>(y, x);
            if (abs(d - median) > mConfig.outlierThreshold) {
                depthPatch.at<float>(y, x) = median;
            }
        }
    }
}

void DepthEnhancer::updateFeatureConfidenceMap(const std::vector<cv::KeyPoint>& features,
                                              const cv::Mat& depth) {
    mFeatureConfidenceMap = Mat::zeros(depth.size(), CV_32F);
    
    // Simplified Gaussian weights
    for (const auto& kp : features) {
        int x = kp.pt.x;
        int y = kp.pt.y;
        float sigma = kp.size;
        
        int radius = sigma * 2;  // Reduced from 3
        for (int dy = -radius; dy <= radius; dy += 2) {  // Step by 2
            for (int dx = -radius; dx <= radius; dx += 2) {
                int px = x + dx;
                int py = y + dy;
                
                if (px >= 0 && px < depth.cols && py >= 0 && py < depth.rows) {
                    float dist2 = dx*dx + dy*dy;
                    float weight = exp(-dist2 / (2 * sigma * sigma));
                    
                    // Fill 2x2 block
                    for (int i = 0; i < 2 && py+i < depth.rows; ++i) {
                        for (int j = 0; j < 2 && px+j < depth.cols; ++j) {
                            mFeatureConfidenceMap.at<float>(py+i, px+j) = 
                                max(mFeatureConfidenceMap.at<float>(py+i, px+j), weight);
                        }
                    }
                }
            }
        }
    }
}

void DepthEnhancer::updateHistory(const DepthFrame& frame) {
    lock_guard<mutex> lock(mMutex);
    
    // Add to general history
    mDepthHistory.push_back(frame);
    if (mDepthHistory.size() > mConfig.maxHistorySize) {
        mDepthHistory.pop_front();
    }
    
    // Add to keyframe history if it's a keyframe
    if (frame.isKeyFrame) {
        mKeyFrameHistory.push_back(frame);
        if (mKeyFrameHistory.size() > mConfig.maxKeyFrames) {
            mKeyFrameHistory.pop_front();
        }
    }
}

void DepthEnhancer::update3DPointPool(const DepthFrame& frame) {
    lock_guard<mutex> lock(mMutex);
    
    // Very sparse sampling for 3D points
    const int step = 20;  // Process every 20th pixel
    
    for (int y = 0; y < frame.depth.rows; y += step) {
        for (int x = 0; x < frame.depth.cols; x += step) {
            float d = frame.depth.at<float>(y, x);
            if (!isValidDepth(d)) continue;
            
            uint64_t key = pointToKey(x, y, d);
            
            auto it = m3DPointPool.find(key);
            if (it == m3DPointPool.end()) {
                // New point
                auto point = make_shared<Point3D>();
                point->position = pixel2Camera(x, y, d);
                point->addObservation(d, frame.frameId, frame.timestamp);
                m3DPointPool[key] = point;
            } else {
                // Update existing point
                it->second->addObservation(d, frame.frameId, frame.timestamp);
            }
        }
    }
    
    // Clean old points
    auto it = m3DPointPool.begin();
    while (it != m3DPointPool.end()) {
        if (it->second->timestamps.back() < frame.timestamp - 5.0) {
            it = m3DPointPool.erase(it);
        } else {
            ++it;
        }
    }
    
    // Limit pool size
    if (m3DPointPool.size() > 10000) {
        // Remove oldest points
        vector<pair<double, uint64_t>> timeKeys;
        for (const auto& kv : m3DPointPool) {
            timeKeys.push_back({kv.second->timestamps.back(), kv.first});
        }
        sort(timeKeys.begin(), timeKeys.end());
        
        // Remove oldest 20%
        int removeCount = m3DPointPool.size() / 5;
        for (int i = 0; i < removeCount; ++i) {
            m3DPointPool.erase(timeKeys[i].second);
        }
    }
}

Eigen::Vector3f DepthEnhancer::pixel2Camera(int x, int y, float depth) {
    float z = depth;
    float x3d = (x - mConfig.cx) * z / mConfig.fx;
    float y3d = (y - mConfig.cy) * z / mConfig.fy;
    return Eigen::Vector3f(x3d, y3d, z);
}

cv::Point2f DepthEnhancer::camera2Pixel(const Eigen::Vector3f& p3d) {
    float x = mConfig.fx * p3d.x() / p3d.z() + mConfig.cx;
    float y = mConfig.fy * p3d.y() / p3d.z() + mConfig.cy;
    return Point2f(x, y);
}

uint64_t DepthEnhancer::pointToKey(int x, int y, float depth) {
    // Quantize to 2cm precision for key generation (reduced precision)
    int depth_cm = depth * 50;
    // Simple spatial hashing with downsampling
    int x_bin = x / 4;
    int y_bin = y / 4;
    return ((uint64_t)x_bin << 40) | ((uint64_t)y_bin << 20) | (uint64_t)depth_cm;
}

float DepthEnhancer::computeNCC(const cv::Mat& patch1, const cv::Mat& patch2) {
    if (patch1.empty() || patch2.empty()) return 0.0f;
    
    // Simple normalized cross correlation
    Scalar mean1 = mean(patch1);
    Scalar mean2 = mean(patch2);
    
    Mat p1_zero = patch1 - mean1[0];
    Mat p2_zero = patch2 - mean2[0];
    
    double numerator = p1_zero.dot(p2_zero);
    double denom1 = sqrt(p1_zero.dot(p1_zero));
    double denom2 = sqrt(p2_zero.dot(p2_zero));
    
    if (denom1 * denom2 == 0) return 0.0f;
    
    float ncc = numerator / (denom1 * denom2);
    return (ncc + 1.0f) / 2.0f; // Normalize to [0, 1]
}

float DepthEnhancer::computeGeometricConsistency(int x, int y, float currentD, float projD,
                                                const cv::Mat& currentDepth, 
                                                const cv::Mat& projectedDepth) {
    // Simple depth consistency check
    float depthDiff = fabs(currentD - projD);
    if (depthDiff > mConfig.outlierThreshold) return 0.0f;
    
    // Exponential weight based on depth difference
    return exp(-depthDiff * depthDiff / 0.01f);
}

void DepthEnhancer::reset() {
    lock_guard<mutex> lock(mMutex);
    mDepthHistory.clear();
    mKeyFrameHistory.clear();
    m3DPointPool.clear();
    mFrameCounter = 0;
}

} // namespace ORB_SLAM3