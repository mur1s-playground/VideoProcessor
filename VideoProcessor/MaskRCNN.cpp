#include "MaskRCNN.h"

#include <opencv2/imgproc.hpp>

#include <sstream>
#include <fstream>

#include "VideoSource.h"

#include "ApplicationGraph.h"
#include "MainUI.h"

#include "Vector2.h"

#include "Logger.h"

void mask_rcnn_init(struct mask_rcnn *mrcnn) {
	string classes_file = "./data/mask_rcnn/mscoco_labels.names";
	ifstream ifs(classes_file.c_str());
	string line;
	while (getline(ifs, line)) mrcnn->net_classes_available.push_back(line);

	String text_graph = "./data/mask_rcnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
	String model_weights = "./data/mask_rcnn/frozen_inference_graph.pb";

	mrcnn->net = readNetFromTensorflow(model_weights, text_graph);
	mrcnn->net.setPreferableBackend(DNN_BACKEND_CUDA);
	mrcnn->net.setPreferableTarget(DNN_TARGET_CUDA);

	mrcnn->net_conf_threshold = 0.5f;
	mrcnn->net_mask_threshold = 0.3f;
	mrcnn->draw_box = false;
	mrcnn->draw_mask = true;
	mrcnn->scale = 1.0f;
	mrcnn->smb_det = nullptr;
}

//INTERNAL HELPERS
void draw_box(struct mask_rcnn* mrcnn, int current_output_id, int class_id, Rect box, Mat& object_mask) {
	if (class_id < (int)mrcnn->net_classes_available.size() && find(mrcnn->net_classes_active.begin(), mrcnn->net_classes_active.end(), mrcnn->net_classes_available[class_id]) != mrcnn->net_classes_active.end()) {
		//Scalar color = Scalar(255, 255, 255, 255);
		if (box.y + box.height > mrcnn->v_src_in->video_height) {
			box.height -= (box.y + box.height - mrcnn->v_src_in->video_height);
		}

		resize(object_mask, object_mask, Size(box.width, box.height));
		Mat mask = (object_mask > mrcnn->net_mask_threshold);

		if ((0 <= box.x && 0 <= box.width && box.x + box.width <= mrcnn->v_src_in->video_width && 0 <= box.y && 0 <= box.height && box.y + box.height <= mrcnn->v_src_in->video_height)) {
			//Mat coloredRoi = color + 0.0f * mrcnn->v_src_in->mats[mrcnn->v_src_in->smb_last_used_id](box);
			//coloredRoi.convertTo(coloredRoi, CV_8UC3);

			mask.convertTo(mask, CV_8U);
			//coloredRoi.copyTo(mrcnn->v_src_out->mats[current_output_id](box), mask);
			mask.copyTo(mrcnn->v_src_out->mats[current_output_id](box), mask);
		}
	}
}

void generate_output(struct mask_rcnn* mrcnn, const vector<Mat>& outs, int in_frame_id) {
	Mat out_detections = outs[0];
	Mat out_masks = outs[1];

	const int num_detections = out_detections.size[2];
	const int num_classes = out_masks.size[1];

	out_detections = out_detections.reshape(1, out_detections.total() / 7);

	int previous_frame = mrcnn->v_src_out->smb_last_used_id;
	int current_frame = (mrcnn->v_src_out->smb_last_used_id + 1) % mrcnn->v_src_out->smb_framecount;

	shared_memory_buffer_try_rw(mrcnn->v_src_out->smb, current_frame, true, 8);
	mrcnn->v_src_out->mats[current_frame].setTo(0);
	
	int saved_detections_count_total;
	if (mrcnn->smb_det != nullptr) {
		shared_memory_buffer_try_rw(mrcnn->smb_det, current_frame, true, 8);
		saved_detections_count_total = mrcnn->smb_det->size / (sizeof(struct vector2<int>) * 2);
	}
	int saved_detections_count = 0;
	for (int i = 0; i < num_detections; ++i) {
		float score = out_detections.at<float>(i, 2);
		if (score > mrcnn->net_conf_threshold) {
			int class_id = static_cast<int>(out_detections.at<float>(i, 1));
			int left = static_cast<int>(mrcnn->v_src_in->mats[mrcnn->v_src_in->smb_last_used_id].cols * out_detections.at<float>(i, 3));
			int top = static_cast<int>(mrcnn->v_src_in->mats[mrcnn->v_src_in->smb_last_used_id].rows * out_detections.at<float>(i, 4));
			int right = static_cast<int>(mrcnn->v_src_in->mats[mrcnn->v_src_in->smb_last_used_id].cols * out_detections.at<float>(i, 5));
			int bottom = static_cast<int>(mrcnn->v_src_in->mats[mrcnn->v_src_in->smb_last_used_id].rows * out_detections.at<float>(i, 6));

			left = max(0, min(left, mrcnn->v_src_in->mats[mrcnn->v_src_in->smb_last_used_id].cols - 1));
			top = max(0, min(top, mrcnn->v_src_in->mats[mrcnn->v_src_in->smb_last_used_id].rows - 1));
			right = max(0, min(right, mrcnn->v_src_in->mats[mrcnn->v_src_in->smb_last_used_id].cols - 1));
			bottom = max(0, min(bottom, mrcnn->v_src_in->mats[mrcnn->v_src_in->smb_last_used_id].rows - 1));
			Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

			/*
			stringstream label_ss;
			label_ss << "input frame_id: " << in_frame_id << " output frame_id: " << current_frame;
			putText(mrcnn->v_src_out->mats[current_frame], label_ss.str(), Point(0, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255,255,255), 1);
			*/

			if (mrcnn->smb_det != nullptr) {
				if (saved_detections_count < saved_detections_count_total) {
					unsigned char* start_pos = &mrcnn->smb_det->p_buf_c[current_frame * saved_detections_count_total * (sizeof(struct vector2<int>) * 2) + saved_detections_count * (sizeof(struct vector2<int>) * 2)];
					struct vector2<int>* sp = (struct vector2<int>*) start_pos;
					sp[0] = struct vector2<int>(top, left);
					sp[1] = struct vector2<int>(bottom, right);
					saved_detections_count++;
				}
			}

			if (mrcnn->draw_box) cv::rectangle(mrcnn->v_src_out->mats[current_frame], box, (255, 255, 255), 3);
			if (mrcnn->draw_mask) {
				Mat object_mask(out_masks.size[2], out_masks.size[3], CV_32F, out_masks.ptr<float>(i, class_id));
				draw_box(mrcnn, current_frame, class_id, box, object_mask);
			}
		}
	}

	if (mrcnn->smb_det != nullptr) {
		unsigned char* start_pos = &mrcnn->smb_det->p_buf_c[current_frame * saved_detections_count_total * (sizeof(struct vector2<int>) * 2) + saved_detections_count * (sizeof(struct vector2<int>) * 2)];
		struct vector2<int>* sp = (struct vector2<int>*) start_pos;
		for (int sd = saved_detections_count; sd < saved_detections_count_total; sd++) {
			sp[0] = struct vector2<int>(-1, -1);
			sp[1] = struct vector2<int>(-1, -1);
			sp += 2;
		}
		shared_memory_buffer_set_time(mrcnn->smb_det, current_frame, shared_memory_buffer_get_time(mrcnn->v_src_in->smb, in_frame_id));
		shared_memory_buffer_release_rw(mrcnn->smb_det, current_frame);
		
		shared_memory_buffer_try_rw(mrcnn->smb_det, mrcnn->smb_det->slots, true, 8);
		mrcnn->smb_det->p_buf_c[mrcnn->smb_det->slots * saved_detections_count_total * (sizeof(struct vector2<int>) * 2) + ((mrcnn->smb_det->slots + 1) * 2)] = current_frame;
		shared_memory_buffer_release_rw(mrcnn->smb_det, mrcnn->smb_det->slots);
	}

	shared_memory_buffer_set_time(mrcnn->v_src_out->smb, current_frame, shared_memory_buffer_get_time(mrcnn->v_src_in->smb, in_frame_id));

	shared_memory_buffer_release_rw(mrcnn->v_src_out->smb, current_frame);
	shared_memory_buffer_try_rw(mrcnn->v_src_out->smb, mrcnn->v_src_out->smb_framecount, true, 8);
	mrcnn->v_src_out->smb->p_buf_c[mrcnn->v_src_out->smb_framecount * mrcnn->v_src_out->video_channels * mrcnn->v_src_out->video_height * mrcnn->v_src_out->video_width + ((mrcnn->v_src_out->smb_framecount + 1) * 2)] = current_frame;
	mrcnn->v_src_out->smb_last_used_id = current_frame;
	shared_memory_buffer_release_rw(mrcnn->v_src_out->smb, mrcnn->v_src_out->smb_framecount);
}
// END INTERNAL HELPERS

DWORD* mask_rcnn_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct mask_rcnn* mrcnn = (struct mask_rcnn*)agn->component;
	
	int last_frame = -1;
	while (agn->process_run) {
		application_graph_tps_balancer_timer_start(agn);

		shared_memory_buffer_try_r(mrcnn->v_src_in->smb, mrcnn->v_src_in->smb_framecount, true, 8);
		//slots																	//rw-locks									      //meta
		int next_frame = mrcnn->v_src_in->smb->p_buf_c[mrcnn->v_src_in->smb_framecount * mrcnn->v_src_in->video_channels * mrcnn->v_src_in->video_height * mrcnn->v_src_in->video_width + ((mrcnn->v_src_in->smb_framecount + 1) * 2)];
		shared_memory_buffer_release_r(mrcnn->v_src_in->smb, mrcnn->v_src_in->smb_framecount);

		if (next_frame != last_frame) {
			shared_memory_buffer_try_r(mrcnn->v_src_in->smb, next_frame, true, 8);
			blobFromImage(mrcnn->v_src_in->mats[next_frame], mrcnn->blob, mrcnn->scale, Size(mrcnn->v_src_in->video_width, mrcnn->v_src_in->video_height), Scalar(), true, false);
			mrcnn->net.setInput(mrcnn->blob);
			
			std::vector<String> out_names(2);
			out_names[0] = "detection_out_final";
			out_names[1] = "detection_masks";
			vector<Mat> outs;
			mrcnn->net.forward(outs, out_names);

			generate_output(mrcnn, outs, next_frame);

			shared_memory_buffer_release_r(mrcnn->v_src_in->smb, next_frame);
			last_frame = next_frame;
		}
		application_graph_tps_balancer_timer_stop(agn);
		application_graph_tps_balancer_sleep(agn);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void mask_rcnn_externalise(struct application_graph_node* agn, string& out_str) {
	struct mask_rcnn* mrcnn = (struct mask_rcnn*)agn->component;

	stringstream s_out;
	s_out << mrcnn->net_conf_threshold << std::endl;
	s_out << mrcnn->net_mask_threshold << std::endl;
	s_out << mrcnn->scale << std::endl;

	for (int i = 0; i < mrcnn->net_classes_active.size(); i++) {
		if (i > 0) {
			s_out << ",";
		}
		s_out << mrcnn->net_classes_active[i];
	}
	s_out << std::endl;
	s_out << mrcnn->draw_box << std::endl;
	s_out << mrcnn->draw_mask << std::endl;

	out_str = s_out.str();
}

void mask_rcnn_load(struct mask_rcnn* mrcnn, ifstream& in_f) {
	mask_rcnn_init(mrcnn);

	std::string line;
	std::getline(in_f, line);
	mrcnn->net_conf_threshold = stof(line);
	std::getline(in_f, line);
	mrcnn->net_mask_threshold = stof(line);
	std::getline(in_f, line);
	mrcnn->scale = stof(line);

	std::getline(in_f, line);
	int start = 0;
	int end = line.find_first_of(",", start);
	while (end != std::string::npos) {
		mrcnn->net_classes_active.push_back(line.substr(start, end - start).c_str());
		start = end + 1;
		end = line.find_first_of(",", start);
	}
	mrcnn->net_classes_active.push_back(line.substr(start, end - start).c_str());

	std::getline(in_f, line);
	mrcnn->draw_box = stoi(line) == 1;
	std::getline(in_f, line);
	mrcnn->draw_mask = stoi(line) == 1;
}

void mask_rcnn_destroy(struct application_graph_node *agn) {
	struct mask_rcnn* mrcnn = (struct mask_rcnn*)agn->component;
	delete mrcnn;
}