#include <opencv2/opencv.hpp>
#include "lsd.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>

// 線情報
struct LineInfo {
    cv::Point2f p1, p2;
    double length; // mm
    double width;  // mm
};

struct Result {
    int blur, nfa;
    int num_lines;
    double total_length;
    double total_width;
    bool use_clahe;
    double clipLimit;
    int tileGrid;
};

static inline cv::Mat apply_clahe(const cv::Mat& gray, double clipLimit = 2.0, int tileGrid = 8) {
    CV_Assert(gray.type() == CV_8UC1);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, cv::Size(tileGrid, tileGrid));
    cv::Mat out;
    clahe->apply(gray, out);
    return out;
}

// 幅検出
double get_line_width(const cv::Mat& gray, const cv::Point2f& p1, const cv::Point2f& p2) {
    cv::Point2f dir = p2 - p1;
    float len = cv::norm(dir);
    if (len == 0) return 0.0;
    dir *= 1.0f / len;

    cv::Point2f normal(-dir.y, dir.x);
    cv::Point2f center = (p1 + p2) * 0.5f;

    auto sample = [&](const cv::Point2f& q)->int {
        int x = std::clamp(cvRound(q.x), 0, gray.cols - 1);
        int y = std::clamp(cvRound(q.y), 0, gray.rows - 1);
        return (int)gray.at<uchar>(y, x);
    };

    // 背景推定（法線±15〜25px の中央値）
    std::vector<int> bg; bg.reserve(2 * (25 - 15 + 1));
    for (int d = 15; d <= 25; ++d) {
        cv::Point2f ppos = center + normal * (float)d;
        cv::Point2f pneg = center - normal * (float)d;
        if (ppos.x >= 0 && ppos.x < gray.cols && ppos.y >= 0 && ppos.y < gray.rows) bg.push_back(sample(ppos));
        if (pneg.x >= 0 && pneg.x < gray.cols && pneg.y >= 0 && pneg.y < gray.rows) bg.push_back(sample(pneg));
    }
    if (bg.empty()) return 0.0;
    std::nth_element(bg.begin(), bg.begin() + bg.size()/2, bg.end());
    double Ibg = (double)bg[bg.size()/2];

    // しきい値（中点＋マージン）
    int Ic = sample(center);
    const int margin = 10;
    double T = 0.5 * (Ibg + Ic);

    // 幅カウント（法線方向）
    const int max_check = 30;
    int width_pos = 0, width_neg = 0;

    for (int d = 1; d <= max_check; ++d) {
        cv::Point2f p = center + normal * (float)d;
        if (p.x < 0 || p.x >= gray.cols || p.y < 0 || p.y >= gray.rows) break;
        if (sample(p) <= T - margin) ++width_pos; else break;
    }
    for (int d = 1; d <= max_check; ++d) {
        cv::Point2f p = center - normal * (float)d;
        if (p.x < 0 || p.x >= gray.cols || p.y < 0 || p.y >= gray.rows) break;
        if (sample(p) <= T - margin) ++width_neg; else break;
    }

    // 中心が線内か救済判定
    bool center_dark = (Ic <= T - margin);
    if (!center_dark) {
        cv::Point2f tangent = dir;
        for (int t = -2; t <= 2; ++t) {
            if (t == 0) continue;
            if (sample(center + tangent * (float)t) <= T - margin) { center_dark = true; break; }
        }
    }
    if (!center_dark && width_pos > 0 && width_neg > 0) center_dark = true;

    return double(width_pos + width_neg + (center_dark ? 1 : 0));
}

// 直線検出（LSD）
std::vector<LineInfo> detect_LSD(const cv::Mat& original, int blur_size, int nfa_thresh,
                                 bool use_clahe = true, double clipLimit = 2.0, int tileGrid = 8) {
    cv::Mat gray;
    if (original.channels() == 3) cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
    else gray = original;

    // CLAHE（幅計測もこの画を使う）
    cv::Mat gray_eq = use_clahe ? apply_clahe(gray, clipLimit, tileGrid) : gray;

    // LSD 入力（平滑化は従来通り）
    cv::Mat proc = gray_eq.clone();
    int k = std::max(1, blur_size) | 1;
    cv::GaussianBlur(proc, proc, cv::Size(k, k), 1.0);

    // LSD 用の double 配列
    std::vector<double> dat(proc.rows * proc.cols);
    for (int y = 0; y < proc.rows; ++y)
        for (int x = 0; x < proc.cols; ++x)
            dat[y * proc.cols + x] = (double)proc.at<uchar>(y, x);

    int n_lines = 0;
    double* lines_data = lsd(&n_lines, dat.data(), proc.cols, proc.rows);

    
    double scale_x = 20.0 / proc.cols;
    double scale_y = 20.0 / proc.rows;

    std::vector<LineInfo> all_lines;
    all_lines.reserve(n_lines);

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
                double width_px = get_line_width(gray_eq, p1, p2);
                double width_mm = width_px * (20.0 / proc.cols) * 10.0;
                if (width_mm > 0.5) width_mm = 0.5;
                all_lines.push_back({p1, p2, length_mm, width_mm});
            }
        }
    }

    std::sort(all_lines.begin(), all_lines.end(),
              [](const LineInfo& a, const LineInfo& b) { return a.length > b.length; });

    return all_lines;
}

// 最適な閾値の判断
Result find_best(const cv::Mat& original,
                 bool use_clahe = true, int tileGrid = 13) {
    std::vector<Result> results;
    results.reserve(5 * 4 * ((100 - 10) / 5 + 1)); 

    const double clipCandidates[] = {1.0, 1.25, 1.5};

    for (double clip : clipCandidates) {
        for (int blur = 1; blur <= 3; blur += 1) {
            for (int nfa = 0; nfa <= 80; nfa += 5) {
                auto lines = detect_LSD(original, 2*blur-1, nfa, use_clahe, clip, tileGrid);
                double total_len = 0.0, total_wid = 0.0;
                for (auto& l : lines) { total_len += l.length; total_wid += l.width; }
                Result r;
                r.blur = 2*blur-1;
                r.nfa = nfa;
                r.num_lines = (int)lines.size();
                r.total_length = total_len;
                r.total_width = total_wid;
                r.use_clahe = use_clahe;
                r.clipLimit = clip;
                r.tileGrid = tileGrid;
                results.push_back(r);
            }
        }
    }

    Result best = {};
    bool found = false;
    for (auto& r : results) {
        if (r.num_lines == 0) continue;
        if (!found ||
            r.num_lines < best.num_lines ||
            (r.num_lines == best.num_lines && r.total_length > best.total_length)) {
            best = r; found = true;
        }
    }
    return found ? best : Result{0,0,0,0.0,0.0,use_clahe,1.0,tileGrid};
}

// 結果の描画・表示
void draw_lines(const cv::Mat& original, const std::vector<LineInfo>& lines, int blur_size,
                bool use_clahe, double clipLimit, int tileGrid) {
    cv::Mat gray;
    if (original.channels() == 3) cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
    else gray = original;

    cv::Mat display = use_clahe ? apply_clahe(gray, clipLimit, tileGrid) : gray;

    int k = std::max(1, blur_size) | 1;
    cv::GaussianBlur(display, display, cv::Size(k, k), 1.0);
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
    const bool USE_CLAHE = true;
    const int  TILE_GRID = 13;

    auto start = std::chrono::high_resolution_clock::now();

    auto best = find_best(original, USE_CLAHE, TILE_GRID);
    if (best.num_lines == 0) {
        std::cerr << "No valid result.\n";
    }

    auto lines = detect_LSD(original, best.blur, best.nfa,
                            best.use_clahe, best.clipLimit, best.tileGrid);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "detect5 [処理時間] " << duration.count() << " ms\n";

    draw_lines(original, lines, best.blur, best.use_clahe, best.clipLimit, best.tileGrid);

    std::cout << "Blur: " << best.blur
              << ", NFA: " << best.nfa
              << ", clipLimit: " << best.clipLimit << '\n';
    std::cout << "Lines: " << best.num_lines << '\n'
              << "Total Length (mm): " << best.total_length << '\n'
              << "Total width (mm): " << best.total_width << '\n';

    return 0;
}

// main
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
