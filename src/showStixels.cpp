#include "showStixels.h"
#include "Stixels.h"

void HSV_to_RGB(const float h, const float s, const float v, int *cr, int *cg, int *cb) {
    const float h_prima = h*360.0f / 60.0f;

    const float c = v*s;
    const float h_mod = fmodf(h_prima, 2.0f);
    const float x = c*(1.0f - fabsf(h_mod - 1.0f));

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;

    if (h_prima >= 0) {
        if (h_prima < 1.0f) {
            r = c;
            g = x;
            b = 0.0f;
        }
        else if (h_prima < 2.0f) {
            r = x;
            g = c;
            b = 0.0f;
        }
        else if (h_prima < 3.0f) {
            r = 0.0f;
            g = c;
            b = x;
        }
        else if (h_prima < 4.0f) {
            r = 0.0f;
            g = x;
            b = c;
        }
        else if (h_prima < 5.0f) {
            r = x;
            g = 0.0f;
            b = c;
        }
        else if (h_prima < 6.0f) {
            r = c;
            g = 0.0f;
            b = x;
        }
    }

    const float m = v - c;
    r = (r + m)*255.0f;
    g = (g + m)*255.0f;
    b = (b + m)*255.0f;
    *cr = (int)r;
    *cg = (int)g;
    *cb = (int)b;

}

void ShowStixels(cv::Mat left_frame, cv::Mat& left_frame_stx, Section* stx, StixelModelParameters model_parameters, int real_cols, int horizon_point, int max_dis_display, float disparity_sky){
    //left_frame_stx.create(left_frame.size(), CV_8UC3);
    left_frame.copyTo(left_frame_stx);
    for (size_t i = 0; i < real_cols; i++) {
        //std::vector<Section> column = stixels.at(i);
        Section prev;
        prev.type = -1;
        bool have_prev = false;
        for (size_t j = 0; j < model_parameters.maxSections; j++) {
            int currentIndex = i*model_parameters.maxSections + j;
            Section sec = stx[currentIndex];
            if (sec.type == -1) {
                break;
            }
            sec.vB = left_frame_stx.rows - 1 - sec.vB;
            sec.vT = left_frame_stx.rows - 1 - sec.vT;

            // If disparity is 0 it is sky
            if (sec.type == OBJECT && sec.disparity < disparity_sky) {
                sec.type = SKY;
            }

            // Sky on top of sky
            if (j > 0) {
                if (!have_prev) {
                    prev = stx[currentIndex - 1];
                    prev.vB = left_frame_stx.rows - 1 - prev.vB;
                    prev.vT = left_frame_stx.rows - 1 - prev.vT;
                }

                if (sec.type == SKY && prev.type == SKY) {
                    sec.vT = prev.vT;
                    have_prev = true;
                }
                else {
                    have_prev = false;
                }
            }

            // If the next segment is a sky, skip current
            if (j + 1 < model_parameters.maxSections) {
                Section next = stx[j + 1];
                if (sec.type == SKY && next.type == SKY) continue;
            }
            // Don't show ground
            if (sec.type != GROUND) {
                const int x = i*model_parameters.columnStep + model_parameters.widthMargin;
                const int y = sec.vT;
                const int width = model_parameters.columnStep;
                int height = sec.vB - sec.vT + 1;

                cv::Mat roi = left_frame_stx(cv::Rect(x, y, width, height));

                // Sky = blue
                int cr = 0;
                int cg = 0;
                int cb = 255;

                // Object = from green to red (far to near)
                if (sec.type == OBJECT) {
                    const float dis = (max_dis_display - sec.disparity) / max_dis_display;
                    float dis_f = dis;
                    if (dis_f < 0.0f) {
                        dis_f = 0.0f;
                    }
                    const float h = dis_f*0.3f;
                    const float s = 1.0f;
                    const float v = 1.0f;

                    HSV_to_RGB(h, s, v, &cr, &cg, &cb);
                }

                cv::Mat color;
                const int top = (roi.rows < 2) ? 0 : 1;
                const int bottom = (roi.rows < 2) ? 0 : 1;
                const int left = 1;
                const int right = 1;

                color.create(roi.rows - top - bottom, roi.cols - left - right, roi.type());
                color.setTo(cv::Scalar(cb, cg, cr));

                cv::Mat color_padded;
                color_padded.create(roi.rows, roi.cols, color.type());

                copyMakeBorder(color, color_padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
                const double alpha = 0.6;
                cv::addWeighted(color_padded, alpha, roi, 1.0 - alpha, 0.0, roi);

            }
        }
    }
    if (horizon_point != -1) {
        // Draw Horizon Line
        int thickness = 2;
        int lineType = 8;
        line(left_frame_stx,
            cv::Point(0, horizon_point),
            cv::Point(left_frame_stx.cols - 1, horizon_point),
            cv::Scalar(0, 0, 0),
            thickness,
            lineType);
    }
}