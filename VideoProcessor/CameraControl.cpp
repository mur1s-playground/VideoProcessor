#include "CameraControl.h"

#include "ApplicationGraph.h"
#include "MainUI.h"

#include <fstream>

#include "Vector2.h"

#include "MaskRCNN.h"

#include "Util.h"
#include "Logger.h"

//matrix detection data image dump
//#include "opencv2/imgcodecs.hpp"

void camera_control_simple_move_inverse(int cam_id, struct vector2<float> src, struct vector2<float> onto);
void camera_control_write_control(int id, struct vector2<int> top_left, struct vector2<int> bottom_right, struct cam_sensors cs);

struct vector2<float> cam_calibration_get_hq(float d_1, float alpha);

void camera_control_calibration_from_matrix(struct camera_control* cc);

void camera_control_init(struct camera_control* cc, int camera_count, string camera_meta_path, string sensors_path, string calibration_path) {
	cc->camera_count = camera_count;

	cc->cam_meta = (struct cam_meta_data*)malloc(sizeof(struct cam_meta_data)*camera_count);
	cc->camera_meta_path = camera_meta_path;
	cc->cam_awareness = (struct cam_awareness*)malloc(cc->camera_count * sizeof(struct cam_awareness));
	//TODO: read real meta data
	float res_o = 0;
	for (int c = 0; c < cc->camera_count; c++) {
		cc->cam_meta[c].resolution[0] = 640;
		cc->cam_meta[c].resolution[1] = 360;
		cc->cam_awareness[c].resolution_offset[0] = 0;
		cc->cam_awareness[c].resolution_offset[1] = res_o;
		res_o += cc->cam_meta[c].resolution[1];
	}
	

	//TODO: get better weights
	float np_w[25] = { 0.0770159,0.0767262,0.0763825,0.0759683,0.0754595,0.0748203,0.0739941,0.0728876,0.0713356,0.0690222,0.065287,0.058668,0.0466034,0.0303119,0.0182474,0.0116283,0.00789316,0.00557978,0.00402777,0.00292125,0.00209508,0.00145584,0.000947069,0.000532801,0.000189098 };

	for (int c = 0; c < cc->camera_count; c++) {
		statistic_angle_denoiser_init(&cc->cam_awareness[c].north_pole, 25);
		statistic_angle_denoiser_set_weights(&cc->cam_awareness[c].north_pole, (float *)&np_w);
		
		statistic_angle_denoiser_init(&cc->cam_awareness[c].horizon, 25);
		statistic_angle_denoiser_set_weights(&cc->cam_awareness[c].horizon, (float *)&np_w);

		cc->cam_awareness[c].detection_history.size = 35;
		cc->cam_awareness[c].detection_history.latest_count = -1;
		cc->cam_awareness[c].detection_history.latest_count = 0;
		cc->cam_awareness[c].detection_history.history = (struct cam_detection*) malloc(cc->cam_awareness[c].detection_history.size * sizeof(struct cam_detection));
		memset(cc->cam_awareness[c].detection_history.history, 0, cc->cam_awareness[c].detection_history.size * sizeof(struct cam_detection));
	}
	
	cc->sensors_path = sensors_path;
	cc->calibration_path = calibration_path;
	logger("calibration_path", calibration_path);

	HANDLE calibration_handle = CreateFileA(cc->calibration_path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (calibration_handle != INVALID_HANDLE_VALUE) {
		const int linelength = 256;
		char read_buf[linelength];
		memset(read_buf, 48, linelength * sizeof(char));
		read_buf[linelength - 1] = 10;

		DWORD dwBytesRead;

		int cam_id = 0;
		while (ReadFile(calibration_handle, read_buf, sizeof(read_buf), &dwBytesRead, NULL) && dwBytesRead > 0) {
			logger("bytes read", (int)dwBytesRead);
			if (dwBytesRead == linelength) {
				memcpy(&cc->cam_awareness[cam_id].calibration, read_buf, sizeof(cam_calibration));

				logger("d_1", cc->cam_awareness[cam_id].calibration.d_1);
				logger("pos_x", cc->cam_awareness[cam_id].calibration.position[0]);
				logger("pos_y", cc->cam_awareness[cam_id].calibration.position[1]);
				logger("pos_z", cc->cam_awareness[cam_id].calibration.position[2]);

				logger("fov_c", cc->cam_awareness[cam_id].calibration.lens_fov[0]);
				logger("fov_t", cc->cam_awareness[cam_id].calibration.lens_fov[1]);

				dwBytesRead = 0;
				cam_id++;
			}
		}

		CloseHandle(calibration_handle);
		
	}

	camera_control_calibration_from_matrix(cc);

	cc->cam_sens = nullptr;
	cc->cam_sens_timestamps = nullptr;
	
	cc->calibration = false;
	cc->position_regression = false;

	cc->vs_cams = nullptr;
	cc->smb_det = nullptr;

	cc->smb_shared_state = nullptr;
	cc->shared_state_size_req = cc->camera_count * sizeof(struct camera_control_shared_state);

	cc->smb_detection_sim = nullptr;
}


void camera_control_on_input_connect(struct application_graph_node* agn, int input_id) {
	struct camera_control* cc = (struct camera_control*)agn->component;

	if (input_id == 0) {
		cc->cam_sens = (struct cam_sensors*) malloc(cc->vs_cams->smb->slots * cc->camera_count * sizeof(struct cam_sensors));
		cc->cam_sens_timestamps = (unsigned long long*) malloc(cc->vs_cams->smb->slots * sizeof(unsigned long long));
	}
}

void camera_control_calibration_from_matrix(struct camera_control * cc) {
	struct cam_calibration_process* ccp = new struct cam_calibration_process[cc->camera_count];

	for (int ca = 0; ca < cc->camera_count; ca++) {
		ccp[ca].calibration_discretization = { 16, 9 };
		cc->cam_awareness[ca].calibration.lens_quantization_size = { 0, 0 };

		struct vector2<float> tmp_vec(0.0f, 0.0f);
		struct cam_detection tmp_cd;
		memset(&tmp_cd, 0, sizeof(struct cam_detection));

		for (int cp_s = 0; cp_s < ccp[ca].calibration_discretization[0] * ccp[ca].calibration_discretization[1]; cp_s++) {
			ccp[ca].calibration_object_a_cd.push_back(pair<struct vector2<float>, struct cam_detection>(tmp_vec, tmp_cd));
		}

		stringstream filename_cal_matrix;
		filename_cal_matrix << "R:\\Cams\\calibration_matrix_" << ca << ".bin";
		size_t out_len = 0.0f;
		util_read_binary(filename_cal_matrix.str(), (unsigned char*)ccp[ca].calibration_object_a_cd.data(), &out_len);

		//matrix detection data image dump
		/*
		Mat scattered = cv::Mat(360, 640, CV_8UC3);
		memset(scattered.data, 0, 360 * 640 * 3);

		for (int i = 0; i < 16 * 9; i++) {
			struct cam_detection* det = &ccp[ca].calibration_object_a_cd[i].second;
			struct vector2<float> det_center = cam_detection_get_center(det);
			det_center[1] -= cc->cam_awareness[ca].resolution_offset[1];
			int y1 = det->y1 - cc->cam_awareness[ca].resolution_offset[1];
			int y2 = det->y2 - cc->cam_awareness[ca].resolution_offset[1];
			int x1 = det->x1 - cc->cam_awareness[ca].resolution_offset[0];
			int x2 = det->x2 - cc->cam_awareness[ca].resolution_offset[0];

			for (int dy = -2; dy < 2; dy++) {
				for (int dx = -2; dx < 2; dx++) {
					scattered.data[((int)det_center[1] + dy) * 640 * 3 + ((int)det_center[0] + dx) * 3 + 1] = 255;
				}
			}

			for (int y = y1; y < y2; y++) {
				scattered.data[y * 640 * 3 + x1 * 3] = 255;
				scattered.data[y * 640 * 3 + x1 * 3 + 1] = 255;
				scattered.data[y * 640 * 3 + x1 * 3 + 2] = 255;

				scattered.data[y * 640 * 3 + x2 * 3] = 255;
				scattered.data[y * 640 * 3 + x2 * 3 + 1] = 255;
				scattered.data[y * 640 * 3 + x2 * 3 + 2] = 255;
			}

			for (int x = x1; x < x2; x++) {
				scattered.data[y1 * 640 * 3 + x * 3] = 255;
				scattered.data[y1 * 640 * 3 + x * 3 + 1] = 255;
				scattered.data[y1 * 640 * 3 + x * 3 + 2] = 255;

				scattered.data[y2 * 640 * 3 + x * 3] = 255;
				scattered.data[y2 * 640 * 3 + x * 3 + 1] = 255;
				scattered.data[y2 * 640 * 3 + x * 3 + 2] = 255;
			}
		}
		filename_cal_matrix << ".png";
		cv::imwrite(triangulation.str(), scattered);
		*/

		if (out_len > 0) {
			int center_row = (int)(ccp[ca].calibration_discretization[1] * 0.5f);
			int center_col = (int)(ccp[ca].calibration_discretization[0] * 0.5f);

			struct vector2<float> center_angles = ccp[ca].calibration_object_a_cd[center_row * 16 + center_col].first;
			struct cam_detection* center = &ccp[ca].calibration_object_a_cd[center_row * 16 + center_col].second;
			struct vector2<float> center_d_center = cam_detection_get_center(center);

			int col_offset = 1;
			if (center_d_center[0] > cc->cam_meta[ca].resolution[0] * 0.5f + cc->cam_awareness[ca].resolution_offset[0]) {
				col_offset = -1;
			}

			struct vector2<float> center_angles_2 = ccp[ca].calibration_object_a_cd[center_row * 16 + center_col + col_offset].first;
			struct cam_detection* center_2 = &ccp[ca].calibration_object_a_cd[center_row * 16 + center_col + col_offset].second;
			struct vector2<float> center_d_center_2 = cam_detection_get_center(center_2);

			logger("center_x", center_d_center[0]);
			logger("center_angles", center_angles[0]);
			logger("center_x_2", center_d_center_2[0]);
			logger("center_angles_2", center_angles_2[0]);

			float l = ((float)cc->cam_meta[ca].resolution[0] * 0.5f + cc->cam_awareness[ca].resolution_offset[0] - center_d_center_2[0])/(center_d_center[0] - center_d_center_2[0]);
			logger("l", l);

			float row_offset = 1;
			if (center_d_center[1] > cc->cam_meta[ca].resolution[1] * 0.5f + cc->cam_awareness[ca].resolution_offset[1]) {
				row_offset = -1;
			}
			struct vector2<float> center_h_angles = ccp[ca].calibration_object_a_cd[(center_row + row_offset) * 16 + center_col].first;
			struct cam_detection* center_h = &ccp[ca].calibration_object_a_cd[(center_row + row_offset) * 16 + center_col].second;
			struct vector2<float> center_h_d_center = cam_detection_get_center(center_h);

			logger("center_y", center_d_center[1]);
			logger("center_angles_h", center_angles[1]);
			logger("center_h_y", center_h_d_center[1]);
			logger("center__h_angles_h", center_h_angles[1]);

			float h = ((float)cc->cam_meta[ca].resolution[1] * 0.5f + cc->cam_awareness[ca].resolution_offset[1] - center_h_d_center[1]) / (center_d_center[1] - center_h_d_center[1]);
			logger("h", h);

			float row_offset_2 = 1;
			if (center_d_center_2[1] > cc->cam_meta[ca].resolution[1] * 0.5f + cc->cam_awareness[ca].resolution_offset[1]) {
				row_offset_2 = -1;
			}
			struct vector2<float> center_h2_angles = ccp[ca].calibration_object_a_cd[(center_row + row_offset_2) * 16 + center_col + col_offset].first;
			struct cam_detection* center_h2 = &ccp[ca].calibration_object_a_cd[(center_row + row_offset_2) * 16 + center_col + col_offset].second;
			struct vector2<float> center_h2_d_center = cam_detection_get_center(center_h2);

			logger("center_x", center_h_d_center[0]);
			logger("center_angles", center_h_angles[0]);
			logger("center_x_2", center_h2_d_center[0]);
			logger("center_angles_2", center_h2_angles[0]);

			float l2 = ((float)cc->cam_meta[ca].resolution[0] * 0.5f + cc->cam_awareness[ca].resolution_offset[0] - center_h2_d_center[0]) / (center_h_d_center[0] - center_h2_d_center[0]);

			logger("l2", l2);

			float h2 = ((float)cc->cam_meta[ca].resolution[1] * 0.5f + cc->cam_awareness[ca].resolution_offset[1] - center_d_center_2[1]) / (center_h2_d_center[1] - center_d_center_2[1]);

			//TMP
			ccp[ca].calibration_object_center_angles = {
				((center_angles_2[0] + l * (center_angles[0] - center_angles_2[0])) + (center_h_angles[0] + l2 * (center_h2_angles[0] - center_h_angles[0]))) * 0.5f,
				((center_h_angles[1] + h * (center_angles[1] - center_h_angles[1])) + ((center_h2_angles[1] + h * (center_angles_2[1] - center_h2_angles[1])))) * 0.5f
			};

			logger("center_angles_fin", ccp[ca].calibration_object_center_angles[0]);
			logger("center_angles_fin_h", ccp[ca].calibration_object_center_angles[1]);

			ccp[ca].calibration_object_center.class_id = 37;
			ccp[ca].calibration_object_center.score = 0.5;
			ccp[ca].calibration_object_center.timestamp = 1337;
			ccp[ca].calibration_object_center.x1 = ((center_2->x1 + l * (center->x1 - center_2->x1)) + (center_h->x1 + l2 * (center_h2->x1 - center_h->x1))) * 0.5f;
			ccp[ca].calibration_object_center.x2 = ((center_2->x2 + l * (center->x2 - center_2->x2)) + (center_h->x2 + l2 * (center_h2->x2 - center_h->x2))) * 0.5f;
			ccp[ca].calibration_object_center.y1 = ((center_h->y1 + h * (center->y1 - center_h->y1)) + (center_h2->y1 + h2 * (center_2->y1 - center_h2->y1))) * 0.5f;
			ccp[ca].calibration_object_center.y2 = ((center_h->y2 + h * (center->y2 - center_h->y2)) + (center_h2->y2 + h2 * (center_2->y2 - center_h2->y2))) * 0.5f;
			
			
			struct vector2<float> center_detection_center = cam_detection_get_center(&ccp[ca].calibration_object_center);
			center_angles = ccp[ca].calibration_object_center_angles;

			std::vector<struct vector2<float>> points;
			std::vector<float>				   values_np;
			std::vector<float>				   values_h;

			float zero_shift_np = 360.0f;
			float min_np = ccp[ca].calibration_object_a_cd[0 * ccp[ca].calibration_discretization[0] + (ccp[ca].calibration_discretization[0] - 1)].first[0];
			for (int r = 1; r < ccp[ca].calibration_discretization[1]; r++) {
				float np = ccp[ca].calibration_object_a_cd[r * ccp[ca].calibration_discretization[0] + (ccp[ca].calibration_discretization[0] - 1)].first[0];
				if (statistic_angle_denoiser_is_left_of(min_np, np)) {
					min_np = np;
				}
			}
			zero_shift_np -= min_np;

			float zero_shift_h = 360.0f;
			float min_h = ccp[ca].calibration_object_a_cd[(ccp[ca].calibration_discretization[1] - 1) * ccp[ca].calibration_discretization[0] + 0].first[1];
			for (int c = 1; c < ccp[ca].calibration_discretization[0]; c++) {
				float h = ccp[ca].calibration_object_a_cd[(ccp[ca].calibration_discretization[1] - 1) * ccp[ca].calibration_discretization[0] + c].first[1];
				if (statistic_angle_denoiser_is_left_of(min_h, h)) {
					min_h = h;
				}
			}
			zero_shift_h -= min_h;

			float avg_fov_w = 0.0f;

			//logger("zero_shift_np", zero_shift_np);
			//logger("zero_shift_h", zero_shift_h);
			for (int r = 0; r < ccp[ca].calibration_discretization[1]; r++) {
				stringstream ss_r;

				float tmp_fov_w = 0.0f;
				float tmp_fov_w_dist = 0.0f;

				for (int c = 0; c < ccp[ca].calibration_discretization[0]; c++) {
					struct vector2<float> det_center = cam_detection_get_center(&ccp[ca].calibration_object_a_cd[r * ccp[ca].calibration_discretization[0] + c].second);
					det_center = { det_center[0] - cc->cam_awareness[ca].resolution_offset[0], det_center[1] - cc->cam_awareness[ca].resolution_offset[1] };

					points.push_back(det_center);

					float v_np = ccp[ca].calibration_object_a_cd[r * ccp[ca].calibration_discretization[0] + c].first[0] + zero_shift_np;
					if (v_np >= 360) v_np -= 360;

					if (c == 0) {
						tmp_fov_w_dist = det_center[0];
						tmp_fov_w = v_np;
					} else if (c == ccp[ca].calibration_discretization[0] - 1) {
						tmp_fov_w_dist = det_center[0] - tmp_fov_w_dist;
						tmp_fov_w -= v_np;
					}

					float v_h = ccp[ca].calibration_object_a_cd[r * ccp[ca].calibration_discretization[0] + c].first[1] + zero_shift_h;
					if (v_h >= 360) v_h -= 360;
					//ss_r << "z_n:" << v_np << " z_h:" << v_h << "\t";


					values_np.push_back(v_np);
					values_h.push_back(v_h);
				}
				//logger(ss_r.str());

				avg_fov_w += (tmp_fov_w / tmp_fov_w_dist) / (float)ccp[ca].calibration_discretization[1];

			}
			avg_fov_w *= cc->cam_meta[ca].resolution[0];
			//logger("avg_fov_w", avg_fov_w);
			cc->cam_awareness[ca].calibration.lens_fov[0] = avg_fov_w;

			float avg_fov_h = 0.0f;

			for (int c = 0; c < ccp[ca].calibration_discretization[0]; c++) {
				float tmp_fov_h = values_h[0 * ccp[ca].calibration_discretization[0] + c] - values_h[(ccp[ca].calibration_discretization[1] - 1) * ccp[ca].calibration_discretization[0] + c];
				float tmp_fov_h_dist = cam_detection_get_center(&ccp[ca].calibration_object_a_cd[(ccp[ca].calibration_discretization[1] - 1) * ccp[ca].calibration_discretization[0] + c].second)[1] - cam_detection_get_center(&ccp[ca].calibration_object_a_cd[(0 * ccp[ca].calibration_discretization[0]) + c].second)[1];

				avg_fov_h += (tmp_fov_h / tmp_fov_h_dist) / (float)ccp[ca].calibration_discretization[0];
			}
			avg_fov_h *= cc->cam_meta[ca].resolution[1];
			//logger("avg_fov_h", avg_fov_h);
			cc->cam_awareness[ca].calibration.lens_fov[1] = avg_fov_h;

			cc->cam_awareness[ca].calibration.lens_north_pole = (struct statistic_unscatter_triangulation_2d*)malloc(sizeof(struct statistic_unscatter_triangulation_2d));
			statistic_unscatter_triangulation_init(cc->cam_awareness[ca].calibration.lens_north_pole, ccp[ca].calibration_discretization, cc->cam_meta[ca].resolution);
			statistic_unscatter_triangulation_calculate(cc->cam_awareness[ca].calibration.lens_north_pole, points, values_np);
			int tries = 10;
			while (!statistic_unscatter_triangulation_approximate_missing(cc->cam_awareness[ca].calibration.lens_north_pole) && tries-- > 0);
			float center_np = statistic_unscatter_triangulation_center_shift_inverse(cc->cam_awareness[ca].calibration.lens_north_pole);

			cc->cam_awareness[ca].calibration.lens_horizon = (struct statistic_unscatter_triangulation_2d*)malloc(sizeof(struct statistic_unscatter_triangulation_2d));
			statistic_unscatter_triangulation_init(cc->cam_awareness[ca].calibration.lens_horizon, ccp[ca].calibration_discretization, cc->cam_meta[ca].resolution);
			statistic_unscatter_triangulation_calculate(cc->cam_awareness[ca].calibration.lens_horizon, points, values_h);
			tries = 10;
			while (!statistic_unscatter_triangulation_approximate_missing(cc->cam_awareness[ca].calibration.lens_horizon) && tries-- > 0);
			float center_horizon = statistic_unscatter_triangulation_center_shift_inverse(cc->cam_awareness[ca].calibration.lens_horizon);
			
			/*
			float x_steps = cc->cam_awareness[ca].calibration.lens_north_pole->dimension[0] / (float)cc->cam_awareness[ca].calibration.lens_north_pole->grid_size[0];
			float x_start = x_steps / 2.0f;

			float y_steps = cc->cam_awareness[ca].calibration.lens_north_pole->dimension[1] / (float)cc->cam_awareness[ca].calibration.lens_north_pole->grid_size[1];
			float y_start = y_steps / 2.0f;
			*/
			/*
			logger("ca", ca);
			for (int r = 0; r < ccp[ca].calibration_discretization[1]; r++) {
				stringstream ss_r;
				for (int c = 0; c < ccp[ca].calibration_discretization[0]; c++) {
					struct vector2<float> det_center = cam_detection_get_center(&ccp[ca].calibration_object_a_cd[r * ccp[ca].calibration_discretization[0] + c].second);
					ss_r << "n:" << ccp[ca].calibration_object_a_cd[r * ccp[ca].calibration_discretization[0] + c].first[0] << " h:" << ccp[ca].calibration_object_a_cd[r * ccp[ca].calibration_discretization[0] + c].first[1] << " x:" << det_center[0] << " y:" << det_center[1] << "\t";
				}
				logger(ss_r.str());
				
			}
			*/
			/*
			for (int r = 0; r < ccp[ca].calibration_discretization[1]; r++) {
				stringstream ss_r2;
				for (int c = 0; c < ccp[ca].calibration_discretization[0]; c++) {
					struct vector2<float> target = { x_start + c * x_steps, y_start + r * y_steps };
					struct vector2<float> det_center = cam_detection_get_center(&ccp[ca].calibration_object_a_cd[r * ccp[ca].calibration_discretization[0] + c].second);
					ss_r2 << "n:" << cc->cam_awareness[ca].calibration.lens_north_pole->data[r * ccp[ca].calibration_discretization[0] + c] << " h:" << cc->cam_awareness[ca].calibration.lens_horizon->data[r * ccp[ca].calibration_discretization[0] + c] << " x:" << target[0] << " y: " << target[1] << "\t";
				}
				logger(ss_r2.str());
			}
			logger("-----------");
			*/

			//cfg
			float calibration_object_diameter = 2.0f; //-> 1m

			struct vector2<float> det_center = cam_detection_get_center(&ccp[ca].calibration_object_center);

			float x_times = (float)cc->cam_meta[ca].resolution[0] * 0.5f / (float)(det_center[0] - ccp[ca].calibration_object_center.x1);

			float a = x_times * calibration_object_diameter * 0.5f;
			float alpha = (cc->cam_awareness[ca].calibration.lens_fov[0] * 0.5f);

			float c = a / sinf(alpha * M_PI / 180.0f);
			cc->cam_awareness[ca].calibration.d_1 = c;

			float h_smoothed = ccp[ca].calibration_object_center_angles[1];
			float pz = h_smoothed * M_PI / (180.0f);

			float np_smoothed = ccp[ca].calibration_object_center_angles[0];

			struct vector3<float>	ray_dir = {
				 sinf((h_smoothed + 90.0f) * M_PI / (2.0f * 90.0f)) * cosf((np_smoothed - 90.0f) * M_PI / (2.0f * 90.0f)),
				 -sinf((h_smoothed + 90.0f) * M_PI / (2.0f * 90.0f)) * sinf((np_smoothed - 90.0f) * M_PI / (2.0f * 90.0f)),
				 cosf((h_smoothed + 90.0f) * M_PI / (2.0f * 90.0f))
			};

			struct vector3<float>	c_pos = -(ray_dir)*c;

			cc->cam_awareness[ca].calibration.position = c_pos;

			logger("cam_id", ca);
			logger("alpha", cc->cam_awareness[ca].north_pole.angle);
			logger("beta", cc->cam_awareness[ca].horizon.angle);
			logger("position_x", cc->cam_awareness[ca].calibration.position[0]);
			logger("position_y", cc->cam_awareness[ca].calibration.position[1]);
			logger("position_z", cc->cam_awareness[ca].calibration.position[2]);
			logger("det_avg_d", (((ccp[ca].calibration_object_center.x2 - ccp[ca].calibration_object_center.x1) + (ccp[ca].calibration_object_center.y2 - ccp[ca].calibration_object_center.y1)) * 0.5f));
			logger("det_dist", cc->cam_awareness[ca].calibration.d_1);
			/*
			if (current_state_slot > -1) {
				ccss[ca].position = cc->cam_awareness[ca].calibration.position;
			}
			*/

		}
	}

}

DWORD* camera_control_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct camera_control* cc = (struct camera_control*)agn->component;

	int saved_detections_count_total = cc->smb_det->size / (sizeof(struct mask_rcnn_detection));

	int last_cameras_frame = 0;
	int last_detection_frame = 0;

	int linelength_sensors = 64;

	int last_state_slot = 0;

	bool calibration_started	= false;
	bool calibration_done		= false;

	struct cam_calibration_process* ccp = nullptr;
	struct statistic_detection_matcher_2d* ccp_sdm2d = nullptr;
	bool had_new_detection_frame = false;

	struct statistic_camera_ray_data scrd;
	statistic_camera_ray_data_init(&scrd, cc);

	struct statistic_detection_matcher_3d sdm3d;
	statistic_detection_matcher_3d_init(&sdm3d, 10, 1000000000, cc, 64, &scrd);

	struct statistic_position_regression spr;
	statistic_position_regression_init(&spr, struct vector3<float>(0.1, 0.1, 0.1), cc, "R:\\Cams\\tmp\\", &scrd, 1000);

	int current_state_slot = -1;

	struct camera_control_shared_state* ccss = nullptr;

	//3D detection sim//
	int last_sim_id = -1;
	//--------------------//

	int tick = 0;

	if (cc->smb_shared_state != nullptr) {
		shared_memory_buffer_try_rw(cc->smb_shared_state, last_state_slot, true, 8);
		ccss = (struct camera_control_shared_state*)&cc->smb_shared_state->p_buf_c[last_state_slot * cc->smb_shared_state->size];
		for (int c = 0; c < cc->camera_count; c++) {
			struct camera_control_shared_state ccss_i;
			memset(&ccss_i, 0, sizeof(struct camera_control_shared_state));
			ccss_i.position = cc->cam_awareness[c].calibration.position;
			ccss_i.fov = cc->cam_awareness[c].calibration.lens_fov;
			memcpy(ccss, &ccss_i, sizeof(struct camera_control_shared_state));
			ccss++;
		}
		shared_memory_buffer_release_rw(cc->smb_shared_state, last_state_slot);
	}


	while (agn->process_run) {
		application_graph_tps_balancer_timer_start(agn);
		
		if (cc->smb_shared_state != nullptr) {
			current_state_slot = (last_state_slot + 1) % cc->smb_shared_state->slots;
			shared_memory_buffer_try_rw(cc->smb_shared_state, current_state_slot, true, 8);
			shared_memory_buffer_try_r(cc->smb_shared_state, last_state_slot, true, 8);
			ccss = (struct camera_control_shared_state*) &cc->smb_shared_state->p_buf_c[current_state_slot * cc->smb_shared_state->size];
			memcpy(ccss, &cc->smb_shared_state->p_buf_c[last_state_slot * cc->smb_shared_state->size], cc->smb_shared_state->size);
			shared_memory_buffer_release_r(cc->smb_shared_state, last_state_slot);
		}

		shared_memory_buffer_try_r(cc->vs_cams->smb, cc->vs_cams->smb_framecount, true, 8);
		int current_cameras_frame = cc->vs_cams->smb->p_buf_c[cc->vs_cams->smb_framecount * cc->vs_cams->video_channels * cc->vs_cams->video_height * cc->vs_cams->video_width + ((cc->vs_cams->smb_framecount + 1) * 2)];
		shared_memory_buffer_release_r(cc->vs_cams->smb, cc->vs_cams->smb_framecount);

		if (current_cameras_frame != last_cameras_frame) {
			if (last_cameras_frame > -1) {
				shared_memory_buffer_try_r(cc->vs_cams->smb, last_cameras_frame, true, 8);
				cc->cam_sens_timestamps[last_cameras_frame] = shared_memory_buffer_get_time(cc->vs_cams->smb, last_cameras_frame);

				HANDLE state_handle = CreateFileA(cc->sensors_path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
				if (state_handle != INVALID_HANDLE_VALUE) {
					if (last_cameras_frame > 0) {
						SetFilePointer(state_handle, last_cameras_frame * linelength_sensors, NULL, FILE_BEGIN);
					}
					
					DWORD  dwBytesRead;
					BYTE   buff[128];
					if (ReadFile(state_handle, buff, sizeof(buff), &dwBytesRead, NULL) && dwBytesRead > 0) {
						if (dwBytesRead >= linelength_sensors) {
							memcpy(&cc->cam_sens[last_cameras_frame * cc->camera_count], buff, cc->camera_count * sizeof(struct cam_sensors));

							for (int c = 0; c < cc->camera_count; c++) {
								statistic_angle_denoiser_update(&cc->cam_awareness[c].north_pole, (float)cc->cam_sens[last_cameras_frame * cc->camera_count + c].compass);					
								statistic_angle_denoiser_update(&cc->cam_awareness[c].horizon, (float)cc->cam_sens[last_cameras_frame * cc->camera_count + c].tilt);
								if (current_state_slot > -1) {
									ccss[c].np_sensor = (float)cc->cam_sens[last_cameras_frame * cc->camera_count + c].compass;
									ccss[c].np_angle = cc->cam_awareness[c].north_pole.angle;
									ccss[c].np_stability = cc->cam_awareness[c].north_pole.angle_stability;

									ccss[c].horizon_sensor = (float)cc->cam_sens[last_cameras_frame * cc->camera_count + c].tilt;
									ccss[c].horizon_angle = cc->cam_awareness[c].horizon.angle;
									ccss[c].horizon_stability = cc->cam_awareness[c].horizon.angle_stability;
								}
							}
						}
					}
					CloseHandle(state_handle);
				}

				shared_memory_buffer_release_r(cc->vs_cams->smb, last_cameras_frame);
			}
			last_cameras_frame = current_cameras_frame;
		}

		shared_memory_buffer_try_r(cc->smb_det, cc->smb_det->slots, true, 8);
		int current_detection_frame = cc->smb_det->p_buf_c[cc->smb_det->slots * saved_detections_count_total * (sizeof(struct mask_rcnn_detection)) + ((cc->smb_det->slots + 1) * 2)];
		shared_memory_buffer_release_r(cc->smb_det, cc->smb_det->slots);

		if (current_detection_frame != last_detection_frame) {
			had_new_detection_frame = true;
			shared_memory_buffer_try_r(cc->smb_det, current_detection_frame, true, 8);
			unsigned char* start_pos = &cc->smb_det->p_buf_c[current_detection_frame * saved_detections_count_total * (sizeof(struct mask_rcnn_detection))];
			struct mask_rcnn_detection* mrd = (struct mask_rcnn_detection*)start_pos;
			unsigned long long detections_time = shared_memory_buffer_get_time(cc->smb_det, current_detection_frame);
			for (int ca = 0; ca < cc->camera_count; ca++) {
				cc->cam_awareness[ca].detection_history.latest_count = 0;
				if (current_state_slot > -1) {
					ccss[ca].latest_detections_used_ct = 0;
				}
			}
			for (int sp = 0; sp < saved_detections_count_total; sp++) {
				if (mrd->score == 0) break;
				int center_y = mrd->y1 + 0.5 * (mrd->y2 - mrd->y1);
				int res_y = 0;
				for (int ca = 0; ca < cc->camera_count; ca++) {
					res_y += cc->cam_meta[ca].resolution[1];
					if (res_y > center_y) {
						cc->cam_awareness[ca].detection_history.latest_count++;
						cc->cam_awareness[ca].detection_history.latest_idx = (cc->cam_awareness[ca].detection_history.latest_idx + 1) % cc->cam_awareness[ca].detection_history.size;
						cc->cam_awareness[ca].detection_history.history[cc->cam_awareness[ca].detection_history.latest_idx].class_id	= mrd->class_id;
						cc->cam_awareness[ca].detection_history.history[cc->cam_awareness[ca].detection_history.latest_idx].score		= mrd->score;
						cc->cam_awareness[ca].detection_history.history[cc->cam_awareness[ca].detection_history.latest_idx].x1			= mrd->x1;
						cc->cam_awareness[ca].detection_history.history[cc->cam_awareness[ca].detection_history.latest_idx].x2			= mrd->x2;
						cc->cam_awareness[ca].detection_history.history[cc->cam_awareness[ca].detection_history.latest_idx].y1			= mrd->y1;
						cc->cam_awareness[ca].detection_history.history[cc->cam_awareness[ca].detection_history.latest_idx].y2			= mrd->y2;
						cc->cam_awareness[ca].detection_history.history[cc->cam_awareness[ca].detection_history.latest_idx].timestamp	= detections_time;
						
						
						if (current_state_slot > -1 && ccss[ca].latest_detections_used_ct < 5) {
							memcpy(&ccss[ca].latest_detections[ccss[ca].latest_detections_used_ct], &cc->cam_awareness[ca].detection_history.history[cc->cam_awareness[ca].detection_history.latest_idx], sizeof(cam_detection));
							ccss[ca].latest_detections_used_ct++;
						}
						
						/*
						logger("new detection");
						logger("cam_id", ca);
						logger("class_id", cc->cam_awareness[ca].detection_history.history[cc->cam_awareness[ca].detection_history.latest_idx].class_id);
						logger("score", cc->cam_awareness[ca].detection_history.history[cc->cam_awareness[ca].detection_history.latest_idx].score);
						logger("timestamp", cc->cam_awareness[ca].detection_history.history[cc->cam_awareness[ca].detection_history.latest_idx].timestamp);
						*/
						break;
					}
				}
				mrd++;
			}
			shared_memory_buffer_release_r(cc->smb_det, current_detection_frame);
			last_detection_frame = current_detection_frame;

			if (!cc->calibration && !cc->position_regression) {
				//if (tick > 200) {
					statistic_detection_matcher_3d_update(&sdm3d, cc, ccss);
				//}
				tick++;
				/*
				for (int d = 0; d < sdm3d.size; d++) {
					if (sdm3d.detections[d].timestamp > 0) {
						logger("------------");
						logger("idx", d);
						logger("class_id", sdm3d.detections[d].class_id);
						logger("position_x", sdm3d.detections[d].position[0]);
						logger("position_y", sdm3d.detections[d].position[1]);
						logger("position_z", sdm3d.detections[d].position[2]);
						logger("timestamp", sdm3d.detections[d].timestamp);
						logger("score", sdm3d.detections[d].score);
						logger("------------");
					}
				}
				*/
			} else if (cc->position_regression) {
				if (spr.t_c < spr.t_samples_count) {
					//statistic_position_regression_update(&spr, cc);
					spr.t_c = spr.t_samples_count;
				} else {
					statistic_position_regression_calculate(&spr);
					cc->position_regression = false;
				}
			}
		}

		//3D detection sim//
		if (cc->smb_detection_sim != nullptr) {
			shared_memory_buffer_try_r(cc->smb_detection_sim, cc->smb_detection_sim->slots, true, 8);
			int current_sim_id = cc->smb_detection_sim->p_buf_c[cc->smb_detection_sim->slots * 7 * sizeof(struct cam_detection_3d) + (cc->smb_detection_sim->slots + 1) * 2];
			shared_memory_buffer_release_r(cc->smb_detection_sim, cc->smb_detection_sim->slots);

			if (current_sim_id != last_sim_id) {
				shared_memory_buffer_try_r(cc->smb_detection_sim, current_sim_id, true, 8);

				int shared_objects = 0;
				if (ccss != nullptr) {
					for (int ca = 0; ca < cc->camera_count; ca++) {
						memset(&ccss[ca].latest_detections_objects, 0, 5 * sizeof(struct vector3<float>));
					}
				}
				int object_count = cc->smb_detection_sim->size / sizeof(struct cam_detection_3d);

				struct cam_detection_3d* tmp_3d_det = (struct cam_detection_3d*)&cc->smb_detection_sim->p_buf_c[current_sim_id * object_count * sizeof(struct cam_detection_3d)];
				for (int sa = 0; sa < object_count; sa++) {
					if (sa / 5 == cc->camera_count) break;
					memcpy(&ccss[sa / 5].latest_detections_3d[sa % 5], &tmp_3d_det[sa], sizeof(struct cam_detection_3d));
				}

				int shared_rays = 0;

				for (int sa = 0; sa < object_count; sa++) {
					if (sa / 5 == cc->camera_count) break;
					ccss[shared_objects / 5].latest_detections_objects[shared_objects % 5] = tmp_3d_det->position;
					for (int r = 0; r < 3; r++) {
						ccss[shared_rays / 15].latest_detections_rays_origin[shared_rays % 15] = tmp_3d_det->ray_position[r];
						ccss[shared_rays / 15].latest_detections_rays[shared_rays % 15] = tmp_3d_det->ray_direction[r];
						shared_rays++;
					}
					shared_objects++;
					tmp_3d_det++;
				}

				shared_memory_buffer_release_r(cc->smb_detection_sim, current_sim_id);
				last_sim_id = current_sim_id;
			}
		}

		if (cc->calibration) {
			int CAM_CALIBRATION_CLASS_ID = 37;
			if (!calibration_started) {
				ccp = new struct cam_calibration_process[cc->camera_count]; //(struct cam_calibration_process*) malloc(cc->camera_count * sizeof(struct cam_calibration_process));
				ccp_sdm2d = (struct statistic_detection_matcher_2d*) malloc(cc->camera_count * sizeof(struct statistic_detection_matcher_2d));
				
				for (int ca = 0; ca < cc->camera_count; ca++) {
					ccp[ca].tick = 0;
					ccp[ca].ccps = CCPS_CALIBRATION_OBJECT_SEARCH;
					
					//TODO: think about ttl
					statistic_detection_matcher_2d_init(&ccp_sdm2d[ca], 4, 1000000000, 10);

					ccp[ca].camera_starting_angles[0] = cc->cam_awareness[ca].north_pole.angle;
					ccp[ca].camera_starting_angles[1] = cc->cam_awareness[ca].horizon.angle;

					ccp[ca].calibration_discretization = { 16, 9 };
					ccp[ca].calibration_object_a_cd_i = -1;

					struct vector2<float> tmp_vec(0.0f, 0.0f);
					struct cam_detection tmp_cd;
					memset(&tmp_cd, 0, sizeof(struct cam_detection));

					for (int cp_s = 0; cp_s < ccp[ca].calibration_discretization[0] * ccp[ca].calibration_discretization[1]; cp_s++) {
						ccp[ca].calibration_object_a_cd.push_back(pair<struct vector2<float>, struct cam_detection>(tmp_vec, tmp_cd));
					}
				}
				calibration_started = true;
			}
			bool calibration_done = true;
			for (int ca = 0; ca < cc->camera_count; ca++) {
				if (ccp[ca].ccps < CCPS_CALIBRATION_DONE) {
					calibration_done = false;
					break;
				}
			}

			if (calibration_done) {

				const int linelength = 256;
				char write_buf[linelength];
				memset(write_buf, 48, linelength * sizeof(char));
				write_buf[linelength - 1] = 10;

				DWORD dwBytesWritten;

				HANDLE calibration_handle = CreateFileA(cc->calibration_path.c_str(),
					FILE_GENERIC_WRITE,         // open for writing
					FILE_SHARE_READ,          // allow multiple readers
					NULL,                     // no security
					OPEN_ALWAYS,              // open or create
					FILE_ATTRIBUTE_NORMAL,    // normal file
					NULL);                    // no attr. template

				for (int ca = 0; ca < cc->camera_count; ca++) {
					/*
					memset(write_buf, 48, linelength * sizeof(char));
					write_buf[linelength - 1] = 10;

					//

					memcpy(write_buf, &cc->cam_awareness[ca].calibration, sizeof(cam_calibration));
					WriteFile(calibration_handle, write_buf, linelength * sizeof(char), &dwBytesWritten, NULL);
					*/

					stringstream filename_cal_matrix;
					filename_cal_matrix << "R:\\Cams\\calibration_matrix_" << ca << ".bin";
					util_write_binary(filename_cal_matrix.str(), (unsigned char *)ccp[ca].calibration_object_a_cd.data(), ccp[ca].calibration_object_a_cd.size() * sizeof(pair<struct vector2<float>, struct cam_detection>));
				}

				CloseHandle(calibration_handle);

				cc->calibration = false;
			}
			
			for (int ca = 0; ca < cc->camera_count; ca++) {
				if (ccp[ca].ccps == CCPS_CALIBRATION_DONE) {
					continue;
				}
				struct cam_awareness* current_awareness = &cc->cam_awareness[ca];
				if (had_new_detection_frame) {
					//logger("camera_id", ca);
					if (ccp[ca].ccps == CCPS_CALIBRATION_OBJECT_SEARCH) {
						//logger("calibration_state", "COS");
						struct cam_detection_history* current_detection_history = &current_awareness->detection_history;
						bool found_object = false;
						if (current_detection_history->latest_count >= 1) {
							int lc_i = current_detection_history->latest_idx;
							for (int lc = 0; lc < current_detection_history->latest_count; lc++) {
								if (current_detection_history->history[lc_i].class_id == CAM_CALIBRATION_CLASS_ID) {
									ccp[ca].ccps = CCPS_CALIBRATION_OBJECT_DETECT_STABLE;
									ccp[ca].tick = 0;
									found_object = true;
									break;
								}
								lc_i--;
								if (lc_i < 0) lc_i += current_detection_history->size;
							}
						} 
						if (!found_object) {
							if (ccp[ca].tick % 10 == 0) {
								float lambda = ccp[ca].tick / 3000.0f;
								float gamma = ccp[ca].tick / 1800.0f;
								float delta = ccp[ca].tick / 1000.0f;
								if (delta > 1.0) delta = 1.0f;
								struct vector2<float> target_angles = {
									ccp[ca].camera_starting_angles[0] + lambda * 360,
									ccp[ca].camera_starting_angles[1] + delta * (((sin(gamma * 3.1415f / (2.0f * 90.0f)) + 1) * 0.5f * 90.0f) - ccp[ca].camera_starting_angles[1])
								};
								camera_control_simple_move_inverse(ca, struct vector2<float>(cc->cam_awareness[ca].north_pole.angle, cc->cam_awareness[ca].horizon.angle), target_angles);
							}
						}
					}
					if (ccp[ca].ccps == CCPS_CALIBRATION_OBJECT_LOST) {
						//logger("calibration_state", "COL");
						struct cam_detection_history* current_detection_history = &current_awareness->detection_history;
						if (current_detection_history->latest_count >= 1) {
							statistic_detection_matcher_2d_update(&ccp_sdm2d[ca], current_detection_history);
						}
						int sm = statistic_detection_matcher_2d_get_stable_match(&ccp_sdm2d[ca], CAM_CALIBRATION_CLASS_ID, 10);
						if (sm > -1) {
							ccp[ca].ccps = ccp[ca].ccps_store;
							ccp[ca].tick = 0;
						} else {
							if (ccp[ca].tick % 10 == 0) {
								camera_control_simple_move_inverse(ca, struct vector2<float>(cc->cam_awareness[ca].north_pole.angle, cc->cam_awareness[ca].horizon.angle), struct vector2<float>(ccp[ca].calibration_object_found_angles[0], ccp[ca].calibration_object_found_angles[1]));
							}
						}
						if (ccp[ca].tick > 120) {
							ccp[ca].ccps = CCPS_CALIBRATION_OBJECT_SEARCH;
							ccp[ca].tick = 0;
							continue;
						}
					}
					if (ccp[ca].ccps == CCPS_CALIBRATION_OBJECT_DETECT_STABLE) {
						//logger("calibration_state", "CDS");
						struct cam_detection_history* current_detection_history = &current_awareness->detection_history;
						if (current_detection_history->latest_count >= 1) {
							statistic_detection_matcher_2d_update(&ccp_sdm2d[ca], current_detection_history);
						}
						int sm = statistic_detection_matcher_2d_get_stable_match(&ccp_sdm2d[ca], CAM_CALIBRATION_CLASS_ID, 10);
						if (sm > -1) {
							memcpy(&ccp[ca].calibration_object_found, &ccp_sdm2d[ca].detections[sm], sizeof(struct cam_detection));
							ccp[ca].calibration_object_found_angles[0] = cc->cam_awareness[ca].north_pole.angle;
							ccp[ca].calibration_object_found_angles[1] = cc->cam_awareness[ca].horizon.angle;
							ccp[ca].ccps = CCPS_CALIBRATION_OBJECT_GET_MATRIX;
							ccp[ca].tick = 0;
						}
						if (ccp[ca].tick > 120) {
							ccp[ca].ccps = CCPS_CALIBRATION_OBJECT_SEARCH;
							ccp[ca].tick = 0;
							continue;
						}
					}
					if (ccp[ca].ccps == CCPS_CALIBRATION_OBJECT_GET_MATRIX) {
						struct cam_detection_history* current_detection_history = &current_awareness->detection_history;
						if (current_detection_history->latest_count >= 1) {
							statistic_detection_matcher_2d_update(&ccp_sdm2d[ca], current_detection_history);
						}
						int sm = statistic_detection_matcher_2d_get_stable_match(&ccp_sdm2d[ca], CAM_CALIBRATION_CLASS_ID, 10);
						if (sm > -1) {
							struct vector2<float> det_center = cam_detection_get_center(&ccp_sdm2d[ca].detections[sm]);

							float det_half_width = (ccp_sdm2d[ca].detections[sm].x2 - ccp_sdm2d[ca].detections[sm].x1) * 0.5f;
							float det_half_height = (ccp_sdm2d[ca].detections[sm].y2 - ccp_sdm2d[ca].detections[sm].y1) * 0.5f;

							struct vector2<float> c_target(cc->cam_meta[ca].resolution[0] * 0.5f + cc->cam_awareness[ca].resolution_offset[0], cc->cam_meta[ca].resolution[1] * 0.5f + cc->cam_awareness[ca].resolution_offset[1]);
							int row = 0;
							int col = 0;
							if (ccp[ca].calibration_object_a_cd_i > -1) {
								float left	= 3 * det_half_width + cc->cam_awareness[ca].resolution_offset[0];
								float right = cc->cam_meta[ca].resolution[0] - 3 * det_half_width + cc->cam_awareness[ca].resolution_offset[0];
								float top = 3 * det_half_height + cc->cam_awareness[ca].resolution_offset[1];
								float bottom = cc->cam_meta[ca].resolution[1] - 3 * det_half_height + cc->cam_awareness[ca].resolution_offset[1];
								
								row = ccp[ca].calibration_object_a_cd_i / ccp[ca].calibration_discretization[0];
								col = ccp[ca].calibration_object_a_cd_i % ccp[ca].calibration_discretization[0];
								if (row % 2 == 1) {
									col = ccp[ca].calibration_discretization[0] - 1 - col;
								}

								c_target = {
									left	+ (col / ((float)ccp[ca].calibration_discretization[0] - 1)) * (right - left),
									top		+ (row / ((float)ccp[ca].calibration_discretization[1] - 1)) * (bottom - top)
								};
							}

							float target_dist = length(det_center - c_target);
							struct vector2<float> target_dims = { (float)(ccp_sdm2d[ca].detections[sm].x2 - ccp_sdm2d[ca].detections[sm].x1),(float)(ccp_sdm2d[ca].detections[sm].y2 - ccp_sdm2d[ca].detections[sm].y1) };

							if (target_dist < length(target_dims) * 0.2f) {
								if (ccp[ca].calibration_object_a_cd_i == -1) {
									memcpy(&ccp[ca].calibration_object_center, &ccp_sdm2d[ca].detections[sm], sizeof(struct cam_detection));
									ccp[ca].calibration_object_center_angles = { cc->cam_awareness[ca].north_pole.angle, cc->cam_awareness[ca].horizon.angle };
								} else {
									struct vector2<float>* target_angles = &ccp[ca].calibration_object_a_cd[row * ccp[ca].calibration_discretization[0] + col].first;
									struct cam_detection* target_detection = &ccp[ca].calibration_object_a_cd[row * ccp[ca].calibration_discretization[0] + col].second;
									memcpy(target_detection, &ccp_sdm2d[ca].detections[sm], sizeof(struct cam_detection));
									target_angles[0][0] = cc->cam_awareness[ca].north_pole.angle;
									target_angles[0][1] = cc->cam_awareness[ca].horizon.angle;
								}
								ccp[ca].calibration_object_a_cd_i++;
								if (ccp[ca].calibration_object_a_cd_i == ccp[ca].calibration_discretization[0] * ccp[ca].calibration_discretization[1]) {
									ccp[ca].ccps = CCPS_CALIBRATION_DONE;
								}
								ccp[ca].tick = 0;
							} else {
								if (ccp[ca].tick % 10 == 0) {
									camera_control_simple_move_inverse(ca, det_center, c_target);
								}
							}
						} else {
							ccp[ca].ccps_store = ccp[ca].ccps;
							ccp[ca].ccps = CCPS_CALIBRATION_OBJECT_LOST;
							ccp[ca].tick = 0;
						}
					}
					ccp[ca].tick++;
				}
			}
			had_new_detection_frame = false;
		}

		if (cc->smb_shared_state != nullptr) {
			shared_memory_buffer_release_rw(cc->smb_shared_state, current_state_slot);
			shared_memory_buffer_try_rw(cc->smb_shared_state, cc->smb_shared_state->slots, true, 8);
			cc->smb_shared_state->p_buf_c[cc->smb_shared_state->slots * cc->smb_shared_state->size + ((cc->smb_shared_state->slots + 1) * 2)] = current_state_slot;
			shared_memory_buffer_release_rw(cc->smb_shared_state, cc->smb_shared_state->slots);
			last_state_slot = current_state_slot;
		}

		application_graph_tps_balancer_timer_stop(agn);
		application_graph_tps_balancer_sleep(agn);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void camera_control_externalise(struct application_graph_node* agn, string& out_str) {
	struct camera_control* cc = (struct camera_control*)agn->component;

	stringstream s_out;
	s_out << cc->camera_count << std::endl;
	s_out << cc->camera_meta_path << std::endl;
	s_out << cc->sensors_path << std::endl;
	s_out << cc->calibration_path << std::endl;

	out_str = s_out.str();
}

void camera_control_load(struct camera_control* cc, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	int camera_count = stoi(line);
	std::getline(in_f, line);
	string m_path = line;
	std::getline(in_f, line);
	string s_path = line;
	std::getline(in_f, line);
	string c_path = line;
	
	camera_control_init(cc, camera_count, m_path, s_path, c_path);
}

void camera_control_destroy(struct application_graph_node* agn) {

}

int camera_control_written = 0;
void camera_control_simple_move_inverse(int cam_id, struct vector2<float> src, struct vector2<float> onto) {
	float dx = onto[0] - src[0];
	float dy = onto[1] - src[1];
	if (abs(dx) > 1) {
		struct cam_control test;
		test.id = cam_id;
		test.ngin = ENGINE_HORIZONTAL;
		test.volt_time = 0.008f;

		float volt_fac = (-1 * (dx > 0) + 1 * (dx < 0));
		
		test.volt[0] = volt_fac * 0.0f;
		for (int v = 1; v < 23; v++) {
			test.volt[v] = volt_fac * 0.5f;
		}
		test.volt[23] = volt_fac * 0.0f;

		test.volt_time_current = 0.0f;

		stringstream filename;
		filename << "R:\\Cams\\control_" << camera_control_written << ".txt";

		HANDLE control_handle = CreateFileA(filename.str().c_str(),
			FILE_GENERIC_WRITE,
			0,
			NULL,
			OPEN_ALWAYS,
			FILE_ATTRIBUTE_NORMAL,
			NULL);

		char buffer[145];
		memset(buffer, 0, 145);
		buffer[144] = 10;
		memcpy(buffer, &test, 112);

		DWORD dwBytesWritten;

		WriteFile(control_handle, buffer, 145 * sizeof(char), &dwBytesWritten, NULL);
		CloseHandle(control_handle);

		camera_control_written++;
	}
	if (abs(dy) > 1) {
		struct cam_control test;
		test.id = cam_id;
		test.ngin = ENGINE_VERTICAL;
		test.volt_time = 0.008f;

		float volt_fac = (-1 * (dy > 0) + 1 * (dy < 0));

		test.volt[0] = volt_fac * 0.0f;
		for (int v = 1; v < 23; v++) {
			test.volt[v] = volt_fac * 1.0f;
		}
		test.volt[23] = volt_fac * 0.0f;

		test.volt_time_current = 0.0f;

		stringstream filename;
		filename << "R:\\Cams\\control_" << camera_control_written << ".txt";

		HANDLE control_handle = CreateFileA(filename.str().c_str(),
			FILE_GENERIC_WRITE,
			0,
			NULL,
			OPEN_ALWAYS,
			FILE_ATTRIBUTE_NORMAL,
			NULL);

		char buffer[145];
		memset(buffer, 0, 145);
		buffer[144] = 10;
		memcpy(buffer, &test, 112);

		DWORD dwBytesWritten;

		WriteFile(control_handle, buffer, 145 * sizeof(char), &dwBytesWritten, NULL);
		CloseHandle(control_handle);

		camera_control_written++;
	}
}

void camera_control_write_control(int id, struct vector2<int> top_left, struct vector2<int> bottom_right, struct cam_sensors cs) {
	int center_x = bottom_right[1] + 0.5 * (bottom_right[1] - top_left[1]);
	int center_y = bottom_right[0] + 0.5 * (bottom_right[0] - top_left[0]);

	int dx = center_x - 320;
	if (abs(dx) > 20) {
		struct cam_control test;
		test.id = id;
		test.ngin = ENGINE_HORIZONTAL;
		test.volt_time = 0.008f;

		float volt_fac = (-1 * (dx < 0) + 1 * (dx > 0))*0.1;

		test.volt[0] = volt_fac * 0.0f;
		test.volt[1] = volt_fac * 1.0f;
		test.volt[2] = volt_fac * 2.0f;
		test.volt[3] = volt_fac * 3.0f;
		test.volt[4] = volt_fac * 4.0f;
		test.volt[5] = volt_fac * 5.0f;
		test.volt[6] = volt_fac * 6.0f;
		test.volt[7] = volt_fac * 7.0f;
		test.volt[8] = volt_fac * 8.0f;
		test.volt[9] = volt_fac * 9.0f;
		test.volt[10] = volt_fac * 10.0f;
		test.volt[11] = volt_fac * 11.0f;
		test.volt[12] = volt_fac * 11.0f;
		test.volt[13] = volt_fac * 10.0f;
		test.volt[14] = volt_fac * 9.0f;
		test.volt[15] = volt_fac * 8.0f;
		test.volt[16] = volt_fac * 7.0f;
		test.volt[17] = volt_fac * 6.0f;
		test.volt[18] = volt_fac * 5.0f;
		test.volt[19] = volt_fac * 4.0f;
		test.volt[20] = volt_fac * 3.0f;
		test.volt[21] = volt_fac * 2.0f;
		test.volt[22] = volt_fac * 1.0f;
		test.volt[23] = volt_fac * 0.0f;

		test.volt_time_current = 0.0f;

		stringstream filename;
		filename << "R:\\Cams\\control_" << camera_control_written << ".txt";

		HANDLE control_handle = CreateFileA(filename.str().c_str(),
			FILE_GENERIC_WRITE, 
			0,         
			NULL,      
			OPEN_ALWAYS,     
			FILE_ATTRIBUTE_NORMAL, 
			NULL);

		char buffer[145];
		memset(buffer, 0, 145);
		buffer[144] = 10;
		memcpy(buffer, &test, 112);

		DWORD dwBytesWritten;

		WriteFile(control_handle, buffer, 145 * sizeof(char), &dwBytesWritten, NULL);
		CloseHandle(control_handle);

		camera_control_written++;
	}
}

float cam_detection_get_center(struct cam_detection* cd, bool horizontal) {
	if (horizontal) {
		return (cd->x1 + 0.5 * (cd->x2 - cd->x1));
	}
	return (cd->y1 + 0.5 * (cd->y2 - cd->y1));
}

struct vector2<float> cam_detection_get_center(struct cam_detection* cd) {
	return struct vector2<float>(cam_detection_get_center(cd, true), cam_detection_get_center(cd, false));
}

struct vector2<float> cam_calibration_get_hq(float d_1, float alpha) {
	float b = d_1;
	float a = b * tanf(alpha * M_PI / (2.0f * 90.0));
	float c = b / cosf(alpha * M_PI / (2.0f * 90.0));
	float p = (a * a)/c;
	float q = c - p;
	float h = sqrtf(p * q);
	return struct vector2<float>(h, q);
}