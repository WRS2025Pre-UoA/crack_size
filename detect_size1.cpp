#include <opencv2/opencv.hpp>
#include "lsd.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>

struct LineInfo {
    cv::Point2f p1, p2;
    double length; // mm
    double width;  // mm
};

struct Result {
    int threshold, blur, nfa;
    int num_lines;
    double total_length;
    double total_width;
};
//幅検出
double get_line_width(const cv::Mat& bin_blur, const cv::Point2f& p1, const cv::Point2f& p2) {
    cv::Point2f dir = p2 - p1;
    float len = cv::norm(dir);
    if (len == 0) return 0.0f;
    dir *= 1.0f / len;

    cv::Point2f normal(-dir.y, dir.x);
    cv::Point2f center = (p1 + p2) * 0.5f;

    int width_pos = 0;
    int width_neg = 0;
    int max_check = 20;

    for (int offset = 1; offset <= max_check; ++offset) {
        cv::Point2f p = center + normal * static_cast<float>(offset);
        if (p.x < 0 || p.x >= bin_blur.cols || p.y < 0 || p.y >= bin_blur.rows) break;
        uchar val = bin_blur.at<uchar>(cvRound(p.y), cvRound(p.x));
        if (val < 255) width_pos++;
        else break;
    }

    for (int offset = 1; offset <= max_check; ++offset) {
        cv::Point2f p = center - normal * static_cast<float>(offset);
        if (p.x < 0 || p.x >= bin_blur.cols || p.y < 0 || p.y >= bin_blur.rows) break;
        uchar val = bin_blur.at<uchar>(cvRound(p.y), cvRound(p.x));
        if (val < 255) width_neg++;
        else break;
    }

    return static_cast<double>(width_pos + width_neg + 1);
}
//直線検出
std::vector<LineInfo> detect_LSD(const cv::Mat& original, int threshold_val, int blur_size, int nfa_thresh) {
    cv::Mat bin,bin_blur;
    cv::threshold(original, bin, threshold_val, 255, cv::THRESH_BINARY);

    bin_blur = bin.clone();

    int kernel = blur_size | 1;
    cv::GaussianBlur(bin_blur, bin_blur, cv::Size(kernel, kernel), 1.5);

    std::vector<double> dat(bin_blur.rows * bin_blur.cols);
    for (int y = 0; y < bin_blur.rows; ++y)
        for (int x = 0; x < bin_blur.cols; ++x)
            dat[y * bin_blur.cols + x] = bin_blur.at<uchar>(y, x);

    int n_lines = 0;
    double* lines_data = lsd(&n_lines, dat.data(), bin_blur.cols, bin_blur.rows);

    double scale_x = 20.0 / bin_blur.cols;
    double scale_y = 20.0 / bin_blur.rows;

    std::vector<LineInfo> all_lines;
    for (int i = 0; i < n_lines; ++i) {
        if (lines_data[i * 7 + 6] > nfa_thresh) {
            cv::Point2f p1(lines_data[i * 7 + 0], lines_data[i * 7 + 1]);
            cv::Point2f p2(lines_data[i * 7 + 2], lines_data[i * 7 + 3]);

            float dx = p2.x - p1.x;
            float dy = p2.y - p1.y;
            float angle_deg = std::fmod(std::abs(std::atan2(dy, dx)) * 180.0f / CV_PI, 180.0f);

            double dx_cm = dx * scale_x;
            double dy_cm = dy * scale_y;
            double length_mm = std::sqrt(dx_cm * dx_cm + dy_cm * dy_cm) * 10.0;

            bool is_angle_ok =
                std::abs(angle_deg - 0) < 5.0 ||
                std::abs(angle_deg - 45) < 5.0 ||
                std::abs(angle_deg - 90) < 5.0;

            if (is_angle_ok && length_mm >= 20.0 && length_mm <= 200.0) {
                double width_px = get_line_width(bin, p1, p2);
                double width_mm = width_px * (20.0 / bin.cols) * 10.0;
                if (width_mm >=0.1 && width_mm < 5)
                all_lines.push_back({p1, p2, length_mm, width_mm});
            }
        }
    }

    std::sort(all_lines.begin(), all_lines.end(),
              [](const LineInfo& a, const LineInfo& b) { return a.length > b.length; });

    return all_lines;
}
//最適な閾値の判断
Result find_best(const cv::Mat& original) {
    std::vector<Result> results;

    
    for (int threshold = 120; threshold <= 180; threshold += 5) {
        for (int nfa = 10; nfa <= 80; nfa += 5) {
            auto lines = detect_LSD(original, threshold, 5, nfa);
            double total_len = 0.0;
            double total_wid = 0.0;
            for (auto& l : lines) {
                total_len += l.length;
                total_wid += l.width;
            }
            results.push_back({threshold, 5, nfa, (int)lines.size(), total_len,total_wid});
        }
    }
    

    Result best = {};
    bool found = false;
    for (auto& r : results) {
        if (r.num_lines == 0) continue;
        if (!found || r.num_lines < best.num_lines ||
            (r.num_lines == best.num_lines && r.total_length > best.total_length)) {
            best = r;
            found = true;
        }
    }

    return found ? best : Result{0, 0, 0, 0, 0.0, 0.0};
}
//結果の描画・表示
void draw_lines(const cv::Mat& original, const std::vector<LineInfo>& lines,
                int threshold_val, int blur_size) {
    cv::Mat display;
    cv::threshold(original, display, threshold_val, 255, cv::THRESH_BINARY);
    int kernel = blur_size | 1;
    cv::GaussianBlur(display, display, cv::Size(kernel, kernel), 1.5);
    cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);

    for (auto& l : lines) {
        cv::line(display, l.p1, l.p2, cv::Scalar(0, 0, 255), 2);
        char text[64];
        snprintf(text, sizeof(text), "L%.1fmm W%.1fmm", l.length, l.width);
        cv::putText(display, text, l.p1 + cv::Point2f(5, -5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("result_image", display);
    cv::waitKey(0);
}


int run_detection(const cv::Mat& original) {
    auto start = std::chrono::high_resolution_clock::now();
    auto best = find_best(original);
    if (best.num_lines == 0) {
        std::cerr << "No valid result.\n";
    }

    auto lines = detect_LSD(original, best.threshold, best.blur, best.nfa);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "detect5 [処理時間] " << duration.count() << " ms\n";
    draw_lines(original, lines, best.threshold, best.blur);

    std::cout << "Threshold: " << best.threshold
              << ", Blur: " << best.blur
              << ", NFA: " << best.nfa << '\n';
    std::cout << "Lines: " << best.num_lines << '\n'
              << "Total Length (mm): " << best.total_length << '\n'
              << "Total width (mm): " << best.total_width << '\n';

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./lsd_app <image_path>" << std::endl;
        return -1;
    }

    cv::Mat original = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (original.empty()) {
        std::cerr << "Image not found: " << argv[1] << std::endl;
        return -1;
    }

    run_detection(original);
    return 0;
}
