#include <opencv2/opencv.hpp>
#include "lsd.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdlib>

struct LineInfo { cv::Point2f p1, p2; double length, width; };

// 幅検出（そのまま）
double get_line_width(const cv::Mat& gray, const cv::Point2f& p1, const cv::Point2f& p2) {
    cv::Point2f dir = p2 - p1;
    float len = cv::norm(dir);
    if (len == 0) return 0.0;
    dir *= 1.0f / len;
    cv::Point2f normal(-dir.y, dir.x), center = (p1 + p2) * 0.5f;

    auto sample = [&](const cv::Point2f& q)->int {
        int x = std::clamp(cvRound(q.x), 0, gray.cols - 1);
        int y = std::clamp(cvRound(q.y), 0, gray.rows - 1);
        return (int)gray.at<uchar>(y, x);
    };

    // 背景推定（法線±15〜25pxの中央値）
    std::vector<int> bg; bg.reserve(2*(25-15+1));
    for (int d=15; d<=25; ++d) {
        cv::Point2f ppos = center + normal*(float)d, pneg = center - normal*(float)d;
        if (ppos.x>=0 && ppos.x<gray.cols && ppos.y>=0 && ppos.y<gray.rows) bg.push_back(sample(ppos));
        if (pneg.x>=0 && pneg.x<gray.cols && pneg.y>=0 && pneg.y<gray.rows) bg.push_back(sample(pneg));
    }
    if (bg.empty()) return 0.0;
    std::nth_element(bg.begin(), bg.begin()+bg.size()/2, bg.end());
    double Ibg = (double)bg[bg.size()/2];

    int Ic = sample(center);
    const int margin = 10;
    double T = 0.5*(Ibg + Ic);

    const int max_check = 30;
    int width_pos=0, width_neg=0;
    for (int d=1; d<=max_check; ++d) {
        cv::Point2f p = center + normal*(float)d;
        if (p.x<0||p.x>=gray.cols||p.y<0||p.y>=gray.rows) break;
        if (sample(p) <= T - margin) ++width_pos; else break;
    }
    for (int d=1; d<=max_check; ++d) {
        cv::Point2f p = center - normal*(float)d;
        if (p.x<0||p.x>=gray.cols||p.y<0||p.y>=gray.rows) break;
        if (sample(p) <= T - margin) ++width_neg; else break;
    }

    bool center_dark = (Ic <= T - margin);
    if (!center_dark) {
        for (int t=-2; t<=2; ++t) if (t!=0 && sample(center + dir*(float)t) <= T - margin) { center_dark = true; break; }
    }
    if (!center_dark && width_pos>0 && width_neg>0) center_dark = true;

    return double(width_pos + width_neg + (center_dark ? 1 : 0));
}

// 前処理：CLAHE のみ
static void preprocess_gray_CLAHE(const cv::Mat& src_gray, cv::Mat& dst_gray,
                                  double clip_limit, int tile_size) {
    double clip = std::max(0.01, clip_limit);
    int t = std::max(1, tile_size);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip, cv::Size(t, t));
    clahe->apply(src_gray, dst_gray);
}

// LSD 検出：CLAHE → ぼかし → LSD
std::vector<LineInfo> detect_LSD(const cv::Mat& original,
                                 int blur_kernel_odd, int nfa_thresh,
                                 double clahe_clip, int clahe_tile) {
    cv::Mat gray;
    if (original.channels()==3) cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
    else gray = original;

    cv::Mat pre;
    preprocess_gray_CLAHE(gray, pre, clahe_clip, clahe_tile);

    int k = (std::max(1, blur_kernel_odd) | 1);
    cv::Mat proc;
    cv::GaussianBlur(pre, proc, cv::Size(k, k), 1.0);

    std::vector<double> dat(proc.rows * proc.cols);
    for (int y=0; y<proc.rows; ++y) {
        const uchar* row = proc.ptr<uchar>(y);
        for (int x=0; x<proc.cols; ++x) dat[y*proc.cols + x] = (double)row[x];
    }

    int n_lines = 0;
    double* lines_data = lsd(&n_lines, dat.data(), proc.cols, proc.rows);

    const double scale_x = 20.0 / proc.cols, scale_y = 20.0 / proc.rows;
    std::vector<LineInfo> lines; lines.reserve(n_lines);

    for (int i=0; i<n_lines; ++i) {
        if (lines_data[i*7 + 6] > nfa_thresh) { // 実装に応じて要確認
            cv::Point2f p1(lines_data[i*7+0], lines_data[i*7+1]);
            cv::Point2f p2(lines_data[i*7+2], lines_data[i*7+3]);

            float dx = p2.x - p1.x, dy = p2.y - p1.y;
            float ang = std::fmod(std::abs(std::atan2(dy, dx))*180.0f/CV_PI, 180.0f);

            double len_mm = std::sqrt((dx*scale_x)*(dx*scale_x) + (dy*scale_y)*(dy*scale_y)) * 10.0;
            bool ang_ok = std::abs(ang-0)<5 || std::abs(ang-45)<5 || std::abs(ang-90)<5;

            if (ang_ok && 20.0<=len_mm && len_mm<=200.0) {
                // 幅は生の gray で計測（必要なら pre に変更可）
                double w_px = get_line_width(gray, p1, p2);
                double w_mm = w_px * (20.0 / proc.cols) * 10.0;
                if (0.1 <= w_mm && w_mm < 5.0) lines.push_back({p1, p2, len_mm, w_mm});
            }
        }
    }
    // 必要なら free(lines_data);
    std::sort(lines.begin(), lines.end(),
              [](const LineInfo& a, const LineInfo& b){ return a.length > b.length; });
    return lines;
}

// 描画：CLAHE → ぼかし のプレビュー上に結果を重畳
static void draw_lines(const cv::Mat& original, const std::vector<LineInfo>& lines,
                       int blur_kernel_odd, int nfa,
                       double clahe_clip, int clahe_tile) {
    cv::Mat gray; if (original.channels()==3) cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY); else gray = original;

    cv::Mat pre; preprocess_gray_CLAHE(gray, pre, clahe_clip, clahe_tile);
    int k = (std::max(1, blur_kernel_odd) | 1);
    cv::Mat display; cv::GaussianBlur(pre, display, cv::Size(k, k), 1.0);
    cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);

    for (const auto& l : lines) {
        cv::line(display, l.p1, l.p2, cv::Scalar(0,0,255), 2);
        char txt[64]; std::snprintf(txt, sizeof(txt), "L%.1fmm W%.1fmm", l.length, l.width);
        cv::putText(display, txt, l.p1 + cv::Point2f(5,-5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
    }

    char hud[160];
    std::snprintf(hud, sizeof(hud),
        "Blur:%d  NFA:%d  CLAHE clip=%.2f  tile=%d  Lines:%zu",
        blur_kernel_odd, nfa, clahe_clip, clahe_tile, lines.size());
    cv::putText(display, hud, {10,25}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255), 2);

    cv::imshow("LSD Result", display);
}

// UI（Blur / NFA / CLAHE_clip / CLAHE_tile）
static void run_ui(const cv::Mat& original) {
    int blur_s = 2;              // 0–15 → kernel = 2*blur_s + 1
    int nfa    = 30;             // 0–200
    int clahe_clip_x100 = 200;   // 1–500 → clip = /100.0（例: 200 → 2.00）
    int clahe_tile      = 8;     // 1–16

    cv::namedWindow("LSD Result", cv::WINDOW_NORMAL);
    cv::resizeWindow("LSD Result", 1200, 800);

    cv::createTrackbar("Blur(odd=2x+1)", "LSD Result", &blur_s, 15);
    cv::createTrackbar("NFA", "LSD Result", &nfa, 200);
    cv::createTrackbar("CLAHE_clip x100", "LSD Result", &clahe_clip_x100, 500);
    cv::createTrackbar("CLAHE_tile", "LSD Result", &clahe_tile, 16);

    int pb=-1, pn=-1, pc=-1, pt=-1;

    while (true) {
        int key = cv::waitKey(10);
        if (key==27 || key=='q' || key=='Q') break;

        int b  = cv::getTrackbarPos("Blur(odd=2x+1)", "LSD Result");
        int nf = cv::getTrackbarPos("NFA", "LSD Result");
        int cl = cv::getTrackbarPos("CLAHE_clip x100", "LSD Result");
        int tl = cv::getTrackbarPos("CLAHE_tile", "LSD Result");

        if (b!=pb || nf!=pn || cl!=pc || tl!=pt) {
            auto t0 = std::chrono::high_resolution_clock::now();

            int kernel = 2*b + 1;
            double clip = std::max(1, cl) / 100.0; // 0回避
            int tile = std::max(1, tl);

            auto lines = detect_LSD(original, kernel, nf, clip, tile);

            auto t1 = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

            draw_lines(original, lines, kernel, nf, clip, tile);
            std::cout << "[UI] Blur=" << kernel
                      << "  NFA=" << nf
                      << "  CLAHE clip=" << clip
                      << "  tile=" << tile
                      << "  Lines=" << lines.size()
                      << "  (" << ms << " ms)" << std::endl;

            pb=b; pn=nf; pc=cl; pt=tl;
        }
    }
    cv::destroyWindow("LSD Result");
}

int main(int argc, char* argv[]) {
    if (argc < 2) { std::cerr << "Usage: ./lsd_app <image_path>\n"; return -1; }
    cv::Mat original = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (original.empty()) { std::cerr << "Image not found: " << argv[1] << "\n"; return -1; }
    run_ui(original);
    return 0;
}
