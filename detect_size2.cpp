#include <opencv2/opencv.hpp>
#include "lsd.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <chrono>

// equalizeHist の強さを重みで調整（0.0=無補正, 1.0=フル適用）
static inline cv::Mat apply_eq_blend(const cv::Mat& gray, double eq_weight) {
    CV_Assert(gray.type() == CV_8UC1);
    if (eq_weight <= 0.0) return gray.clone();
    if (eq_weight >= 1.0) {
        cv::Mat he; cv::equalizeHist(gray, he); return he;
    }
    cv::Mat he, out;
    cv::equalizeHist(gray, he);
    cv::addWeighted(gray, 1.0 - eq_weight, he, eq_weight, 0.0, out);
    return out;
}

static inline float cross2(const cv::Point2f& u, const cv::Point2f& v) {
    return u.x * v.y - u.y * v.x;
}
//  構造体
struct LineInfo {
    cv::Point2f p1, p2;
    double length; // mm
    double width;  // mm
};

struct Result {
    int blur;
    int nfa;
    int num_lines;
    double total_length;
    double total_width;
    double eq_weight; // equalizeHist ブレンド強度
};

struct LW { // 戻り値用：最長線の長さ・幅
    double length_mm;
    double width_mm;
};
//  重複チェック
bool check_Duplicate(const cv::Point2f& a1, const cv::Point2f& a2,
                     const cv::Point2f& b1, const cv::Point2f& b2) {
    double angle = CV_PI / 36.0; // 5度
    double distance = 10.0;      // 画素距離しきい値

    cv::Point2f dirA = a2 - a1;
    float lenA = cv::norm(dirA);
    if (lenA == 0) return false;
    dirA *= 1.0f / lenA;

    float proj1 = (b1 - a1).dot(dirA);
    float proj2 = (b2 - a1).dot(dirA);
    float min_proj = std::min(proj1, proj2);
    float max_proj = std::max(proj1, proj2);
    float margin = 10.0f;

    if (max_proj < -margin || min_proj > lenA + margin) return false;

    float dist1 = std::abs(cross2((b1 - a1), dirA));
    float dist2 = std::abs(cross2((b2 - a1), dirA));
    if (dist1 > distance || dist2 > distance) return false;

    cv::Point2f dirB = b2 - b1;
    float angleA = std::atan2(dirA.y, dirA.x);
    float angleB = std::atan2(dirB.y, dirB.x);
    float diff = std::abs(angleA - angleB);
    diff = std::fmod(diff, static_cast<float>(CV_PI * 2));
    if (diff > CV_PI) diff = static_cast<float>(CV_PI * 2) - diff;
    if (diff > CV_PI / 2) diff = static_cast<float>(CV_PI) - diff;

    return diff < angle;
}
//  幅検出
double get_line_width(const cv::Mat& gray_img, const cv::Point2f& p1, const cv::Point2f& p2) {
    cv::Point2f dir = p2 - p1;
    float len = cv::norm(dir);
    if (len == 0) return 0.0;
    dir *= 1.0f / len;

    cv::Point2f normal(-dir.y, dir.x);
    cv::Point2f center = (p1 + p2) * 0.5f;

    auto sample = [&](const cv::Point2f& q)->int {
        int x = std::clamp(cvRound(q.x), 0, gray_img.cols - 1);
        int y = std::clamp(cvRound(q.y), 0, gray_img.rows - 1);
        return (int)gray_img.at<uchar>(y, x);
    };

    // 背景推定（法線±15〜25px の中央値）
    std::vector<int> bg; bg.reserve(2 * (25 - 15 + 1));
    for (int d = 15; d <= 25; ++d) {
        cv::Point2f ppos = center + normal * (float)d;
        cv::Point2f pneg = center - normal * (float)d;
        if (ppos.x >= 0 && ppos.x < gray_img.cols && ppos.y >= 0 && ppos.y < gray_img.rows) bg.push_back(sample(ppos));
        if (pneg.x >= 0 && pneg.x < gray_img.cols && pneg.y >= 0 && pneg.y < gray_img.rows) bg.push_back(sample(pneg));
    }
    if (bg.empty()) return 0.0;
    std::nth_element(bg.begin(), bg.begin() + bg.size()/2, bg.end());
    double Ibg = (double)bg[bg.size()/2];

    // しきい値（中点＋マージン）
    int Ic = sample(center);
    if (Ic > Ibg) return 0.0;

    const int margin = 10;
    double T = 0.5 * (Ibg + Ic);

    // 幅カウント（法線方向）
    const int max_check = 30;
    int width_pos = 0, width_neg = 0;

    for (int d = 1; d <= max_check; ++d) {
        cv::Point2f p = center + normal * (float)d;
        if (p.x < 0 || p.x >= gray_img.cols || p.y < 0 || p.y >= gray_img.rows) break;
        if (sample(p) <= T - margin) ++width_pos; else break;
    }
    for (int d = 1; d <= max_check; ++d) {
        cv::Point2f p = center - normal * (float)d;
        if (p.x < 0 || p.x >= gray_img.cols || p.y < 0 || p.y >= gray_img.rows) break;
        if (sample(p) <= T - margin) ++width_neg; else break;
    }

    // 中心救済
    bool center_dark = (Ic <= T - margin);
    if (!center_dark) {
        for (int t = -2; t <= 2; ++t) {
            if (t == 0) continue;
            if (sample(center + dir * (float)t) <= T - margin) { center_dark = true; break; }
        }
    }
    if (!center_dark && width_pos > 0 && width_neg > 0) center_dark = true;

    return double(width_pos + width_neg + (center_dark ? 1 : 0));
}
//  直線検出
std::vector<LineInfo> detect_LSD(const cv::Mat& original,
                                 int blur_size,
                                 int nfa_thresh,
                                 double eq_weight) {
    cv::Mat gray;
    if (original.channels() == 3) cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
    else gray = original.clone();

    // 検出のためのコントラスト（eq_weight でブレンド）
    cv::Mat gray_eq = apply_eq_blend(gray, eq_weight);

    // LSD 入力用に平滑化
    cv::Mat proc = gray_eq.clone();
    int k = std::max(1, blur_size) | 1;
    cv::GaussianBlur(proc, proc, cv::Size(k, k), 1.0);

    // LSD 用 double 配列
    std::vector<double> dat(proc.rows * proc.cols);
    for (int y = 0; y < proc.rows; ++y)
        for (int x = 0; x < proc.cols; ++x)
            dat[y * proc.cols + x] = (double)proc.at<uchar>(y, x);

    int n_lines = 0;
    double* lines_data = lsd(&n_lines, dat.data(), proc.cols, proc.rows);

    // 画素→mm のスケール（画像全体が 20 cm × 20 cm を想定して mm へ）
    const double scale_x = 20.0 / proc.cols;  // cm/px
    const double scale_y = 20.0 / proc.rows;  // cm/px

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

            // 角度：0°, 45°, 90° ±5°、長さ：20–200 mm
            bool is_angle_ok =
                std::abs(angle_deg - 0) < 5.0 ||
                std::abs(angle_deg - 45) < 5.0 ||
                std::abs(angle_deg - 90) < 5.0;

            if (is_angle_ok && length_mm >= 20.0 && length_mm <= 200.0) {
                // 重複チェック
                bool is_dup = false;
                for (const auto& ex : all_lines) {
                    if (check_Duplicate(p1, p2, ex.p1, ex.p2) ||
                        check_Duplicate(ex.p1, ex.p2, p1, p2)) {
                        is_dup = true;
                        break;
                    }
                }
                if (is_dup) continue;

                // ★ 幅は “元画像” で測る方が厳密（推奨）
                double width_px = get_line_width(gray, p1, p2);
                double width_mm = width_px * (20.0 / proc.cols) * 10.0;
                if (width_mm > 0.5) width_mm = 0.5; // 必要なら調整

                all_lines.push_back({p1, p2, length_mm, width_mm});
            }
        }
    }

    // 表示は長さ降順
    std::sort(all_lines.begin(), all_lines.end(),
              [](const LineInfo& a, const LineInfo& b) { return a.length > b.length; });

    return all_lines;
}
//  最適化（本数多い→総延長）＋ eq_weight を探索
Result find_best(const cv::Mat& original) {
    std::vector<Result> results;
    results.reserve(4 * 3 * ((80 - 0)/5 + 1));

    // 探索する equalizeHist ブレンド強度
    const double eq_weights[] = {0.0, 0.3, 0.6, 1.0};

    for (double ew = 0.0; ew <=0.1; ew += 0.02) {
        for (int blur = 1; blur <= 3; blur += 1) {
            for (int nfa = 0; nfa <= 80; nfa += 5) {
                auto lines = detect_LSD(original, 2*blur-1, nfa, ew);
                double total_len = 0.0, total_wid = 0.0;
                for (auto& l : lines) { total_len += l.length; total_wid += l.width; }
                results.push_back(Result{2*blur-1, nfa, (int)lines.size(), total_len, total_wid, ew});
            }
        }
    }

    Result best{}; bool found = false;
    for (auto& r : results) {
        if (r.num_lines == 0) continue;
        if (!found ||
            r.num_lines < best.num_lines ||
            (r.num_lines == best.num_lines && r.total_length > best.total_length)) {
            best = r; found = true;
        }
    }
    if (!found) return Result{1, 0, 0, 0.0, 0.0, 0.0};
    return best;
}
//  描画（最終 eq_weight を反映）
void draw_lines(const cv::Mat& original, const std::vector<LineInfo>& lines, int blur_size, double eq_weight) {
    cv::Mat gray;
    if (original.channels() == 3) cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
    else gray = original.clone();

    // 表示も最終 eq_weight を反映
    cv::Mat display = apply_eq_blend(gray, eq_weight);
    int k = std::max(1, blur_size) | 1;
    cv::GaussianBlur(display, display, cv::Size(k, k), 1.0);
    cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);

    for (auto& l : lines) {
        cv::line(display, l.p1, l.p2, cv::Scalar(0, 0, 255), 2);
        char text[64];
        std::snprintf(text, sizeof(text), "L%.1fmm W%.1fmm", l.length, l.width);
        cv::putText(display, text, l.p1 + cv::Point2f(5, -5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("result_image", display);
    cv::waitKey(0);
}
//  実行：最長線の長さ・幅を戻り値で返す
LW run_detection(const cv::Mat& original) {
    auto start = std::chrono::high_resolution_clock::now();

    auto best = find_best(original);
    auto lines = detect_LSD(original, best.blur, best.nfa, best.eq_weight);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "detect5 [処理時間] " << duration.count() << " ms\n";

    draw_lines(original, lines, best.blur, best.eq_weight);

    std::cout << "Blur: " << best.blur
              << ", NFA: " << best.nfa
              << ", eq_weight: " << best.eq_weight << '\n';
    std::cout << "検出された直線数: " << lines.size() << '\n';

    // 各線の 長さ→幅 を交互に出力
    std::cout << std::fixed << std::setprecision(2);
    for (size_t i = 0; i < lines.size(); ++i) {
        std::cout << "Line " << (i + 1)
                  << " → Length: " << lines[i].length << " mm"
                  << ", Width: "  << lines[i].width  << " mm\n";
    }

    if (!lines.empty()) {
        return LW{ lines.front().length, lines.front().width };
    } else {
        return LW{ 0.0, 0.0 };
    }
}
//  main
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
