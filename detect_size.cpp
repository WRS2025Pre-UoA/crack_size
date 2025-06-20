#include <opencv2/opencv.hpp>
#include "lsd.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>


cv::Mat original;
cv::Mat img;
int n_lines;
double* lines = nullptr;
int clahe_clip;
int blur_size;
int nfa_thresh;

// 重複判定関数
bool check_Duplicate(const cv::Point2f& a1, const cv::Point2f& a2, const cv::Point2f& b1, const cv::Point2f& b2)
{
    double angle = CV_PI / 36.0;
    double distance = 10.0;
    cv::Point2f dirA = a2 - a1;
    float lenA = cv::norm(dirA);
    if (lenA == 0) return false;
    dirA *= 1.0f / lenA;

    float proj1 = (b1 - a1).dot(dirA);
    float proj2 = (b2 - a1).dot(dirA);

    float min_proj = std::min(proj1, proj2);
    float max_proj = std::max(proj1, proj2);

    float margin = 10.0;  
    if (max_proj < -margin || min_proj > lenA + margin) return false;

    float dist1 = std::abs((b1 - a1).cross(dirA));
    float dist2 = std::abs((b2 - a1).cross(dirA));
    if (dist1 > distance || dist2 > distance) return false;

    cv::Point2f dirB = b2 - b1;
    float angleA = atan2(dirA.y, dirA.x);
    float angleB = atan2(dirB.y, dirB.x);
    float diff = std::abs(angleA - angleB);
    diff = std::fmod(diff, static_cast<float>(CV_PI * 2));
    if (diff > CV_PI) diff = static_cast<float>(CV_PI * 2) - diff;
    if (diff > CV_PI / 2) diff = static_cast<float>(CV_PI) - diff;

    /*std::cout << "[debug] proj: " << min_proj << "～" << max_proj
              << " / dist: " << dist1 << "," << dist2
              << " / angle_diff: " << diff << std::endl;*/

    return diff < angle;
}

// LSD実行
std::pair<int, double> update_lsd()
{
    cv::Mat enhanced;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE((double)clahe_clip / 10.0, cv::Size(8, 8));
    clahe->apply(original, enhanced);

    int kernel = blur_size | 1;
    cv::GaussianBlur(enhanced, enhanced, cv::Size(kernel, kernel), 1.5);
    img = enhanced.clone();

    double* dat = new double[img.rows * img.cols];
    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
            dat[y * img.cols + x] = img.at<uchar>(y, x);
    if (lines) delete[] lines;
    lines = lsd(&n_lines, dat, img.cols, img.rows);
    delete[] dat;

    double scale_x = 20.0 / img.cols;
    double scale_y = 20.0 / img.rows;

    struct LineInfo {
        cv::Point2f p1, p2;
        double length;
    };

    std::vector<LineInfo> all_lines;
    for (int i = 0; i < n_lines; i++) {
        if (lines[i * 7 + 6] > nfa_thresh) {
            cv::Point2f p1(lines[i * 7 + 0], lines[i * 7 + 1]);
            cv::Point2f p2(lines[i * 7 + 2], lines[i * 7 + 3]);
            double dx = (p2.x - p1.x) * scale_x;
            double dy = (p2.y - p1.y) * scale_y;
            double len = std::sqrt(dx * dx + dy * dy);
            if (len < 15.0) {
                all_lines.push_back({p1, p2, len});
            }
        }
    }

    std::sort(all_lines.begin(), all_lines.end(), [](const LineInfo& a, const LineInfo& b) {
        return a.length > b.length;
    });

    std::vector<LineInfo> filtered_lines;
    for (const auto& l : all_lines) {
        bool dup = false;
        for (const auto& existing : filtered_lines) {
            if (check_Duplicate(l.p1, l.p2, existing.p1, existing.p2)) {
                dup = true;
                break;
            }
        }
        if (!dup) filtered_lines.push_back(l);
    }

    if (lines) delete[] lines;
    n_lines = filtered_lines.size();
    lines = new double[n_lines * 7];
    for (int i = 0; i < n_lines; i++) {
        const auto& l = filtered_lines[i];
        lines[i * 7 + 0] = l.p1.x;
        lines[i * 7 + 1] = l.p1.y;
        lines[i * 7 + 2] = l.p2.x;
        lines[i * 7 + 3] = l.p2.y;
        lines[i * 7 + 4] = l.length;
        lines[i * 7 + 5] = 1.0;
        lines[i * 7 + 6] = nfa_thresh + 1;
    }

    double total_length = 0.0;
    for (const auto& l : filtered_lines) total_length += l.length;
    
    return {static_cast<int>(filtered_lines.size()), total_length};
}

// 描画関数
void draw_best_result()
{
    std::cout << "[DEBUG] n_lines = " << n_lines << std::endl;

    cv::Mat enhanced;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE((double)clahe_clip / 10.0, cv::Size(8, 8));
    clahe->apply(original, enhanced);

    int kernel = blur_size | 1;
    cv::GaussianBlur(enhanced, enhanced, cv::Size(kernel, kernel), 1.5);
    img = enhanced.clone();
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

    double scale_x = 20.0 / img.cols;
    double scale_y = 20.0 / img.rows;

    for (int i = 0; i < n_lines; i++) {
        cv::Point2f p1(lines[i * 7 + 0], lines[i * 7 + 1]);
        cv::Point2f p2(lines[i * 7 + 2], lines[i * 7 + 3]);

        
        cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 2);

        double dx = (p2.x - p1.x) * scale_x;
        double dy = (p2.y - p1.y) * scale_y;
        double len = std::sqrt(dx * dx + dy * dy);
        char text[32];
        snprintf(text, sizeof(text), "%.1f", len);
        cv::putText(img, text, p1 + cv::Point2f(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("result_image", img);
}


int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: ./lsd_app <image_path>" << std::endl;
        return -1;
    }

    original = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (original.empty()) {
        std::cerr << "Image not found: " << argv[1] << std::endl;
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    struct Result {
        int clahe, blur, nfa;
        int num_lines;
        double total_length;
    };

    std::vector<Result> results;
    //int clahe = 20;
    for (int blur = 3; blur <= 5; blur += 2) {
        for (int clahe = 10; clahe <= 30; clahe += 10){
            for (int nfa = 80; nfa <= 120; nfa += 5) {
                ::clahe_clip = clahe;
                ::blur_size = blur;
                ::nfa_thresh = nfa;

                auto [num_lines, total_len] = update_lsd();
                results.push_back({clahe, blur, nfa, num_lines, total_len});
            }
        }
    }

    bool found = false;
    Result best = results[0];
    for (const auto& r : results) {
        if (r.num_lines == 0) continue;
        if (r.num_lines < best.num_lines ||
            (r.num_lines == best.num_lines && r.total_length > best.total_length)) {
            best = r;
            found = true;
        }
    }

    if (!found) {
        std::cerr << "No valid result.\n";
        return -1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "detect3 [処理時間] " << duration.count() << " ms\n";

    clahe_clip = best.clahe;
    blur_size = best.blur;
    nfa_thresh = best.nfa;
    update_lsd();

    std::cout << "[Best Parameters]\n";
    std::cout << "CLAHE: " << clahe_clip << ", Blur: " << blur_size << ", NFA: " << nfa_thresh << "\n";
    std::cout << "Lines: " << best.num_lines << ", Total Length: " << best.total_length << std::endl;

    cv::namedWindow("result_image", cv::WINDOW_AUTOSIZE);
    draw_best_result();
    cv::waitKey(0);

    if (lines) delete[] lines;
    return 0;
}
