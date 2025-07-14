 #include <opencv2/opencv.hpp>
#include "lsd.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

struct LineInfo {
    cv::Point2f p1, p2;
    double length; // mm
    double width;  // mm
};

struct Result {
    int threshold, blur, nfa;
    int num_lines;
    double total_length;
};
//幅検出
double get_line_width(const cv::Mat& bin, const cv::Point2f& p1, const cv::Point2f& p2) {
    cv::Point2f dir = p2 - p1;
    float len = cv::norm(dir);
    if (len == 0) return 0.0f;
    dir *= 1.0f / len;

    cv::Point2f normal(-dir.y, dir.x);
    cv::Point2f center = (p1 + p2) * 0.5f;

    int width_pos = 0;
    int width_neg = 0;
    int max_check = 20;

    // 正方向（normal）
    for (int offset = 1; offset <= max_check; ++offset) {
        cv::Point2f p = center + normal * static_cast<float>(offset);
        if (p.x < 0 || p.x >= bin.cols || p.y < 0 || p.y >= bin.rows) break;
        uchar val = bin.at<uchar>(cvRound(p.y), cvRound(p.x));
        if (val < 255) width_pos++;
        else break;
    }

    // 負方向（-normal）
    for (int offset = 1; offset <= max_check; ++offset) {
        cv::Point2f p = center - normal * static_cast<float>(offset);
        if (p.x < 0 || p.x >= bin.cols || p.y < 0 || p.y >= bin.rows) break;
        uchar val = bin.at<uchar>(cvRound(p.y), cvRound(p.x));
        if (val < 255) width_neg++;
        else break;
    }

    // 中心ピクセルを含めて合計
    return static_cast<double>(width_pos + width_neg + 1);
}


//長さ検出
std::vector<LineInfo> detect_LSD(const cv::Mat& original, int threshold_val, int blur_size, int nfa_thresh) {
    cv::Mat bin;
    cv::threshold(original, bin, threshold_val, 255, cv::THRESH_BINARY);
    int kernel = blur_size | 1;
    cv::GaussianBlur(bin, bin, cv::Size(kernel, kernel), 1.5);

    std::vector<double> dat(bin.rows * bin.cols);
    for (int y = 0; y < bin.rows; ++y)
        for (int x = 0; x < bin.cols; ++x)
            dat[y * bin.cols + x] = bin.at<uchar>(y, x);

    int n_lines = 0;
    double* lines_data = lsd(&n_lines, dat.data(), bin.cols, bin.rows);

    double scale_x = 20.0 / bin.cols;
    double scale_y = 20.0 / bin.rows;

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
                all_lines.push_back({p1, p2, length_mm, width_mm});
            }
        }
    }

    std::sort(all_lines.begin(), all_lines.end(),
              [](const LineInfo& a, const LineInfo& b) { return a.length > b.length; });

    return all_lines;
}

//閾値の確定
Result find_best(const cv::Mat& original) {
    std::vector<Result> results;

    for (int blur = 5; blur <= 5; blur += 2) {
        for (int threshold = 120; threshold <= 180; threshold += 5) {
            for (int nfa = 80; nfa <= 120; nfa += 5) {
                auto lines = detect_LSD(original, threshold, blur, nfa);
                double total_len = 0.0;
                for (auto& l : lines) total_len += l.length;
                results.push_back({threshold, blur, nfa, (int)lines.size(), total_len});
            }
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

    return found ? best : Result{0, 0, 0, 0, 0.0};
}
//直線の描画と画像の表示
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

//画像の受け取りと処理の実行
int run_detection(const cv::Mat& original) {
    auto start = std::chrono::high_resolution_clock::now();
    auto best = find_best(original);
    if (best.num_lines == 0) {
        std::cerr << "No valid result.\n";
        return -1;
    }

    auto lines = detect_LSD(original, best.threshold, best.blur, best.nfa);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "detect4 [処理時間] " << duration.count() << " ms\n";
    draw_lines(original, lines, best.threshold, best.blur);

    std::cout << "Threshold: " << best.threshold
              << ", Blur: " << best.blur
              << ", NFA: " << best.nfa << '\n';
    std::cout << "Lines: " << best.num_lines
              << ", Total Length (mm): " << best.total_length << '\n';

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
