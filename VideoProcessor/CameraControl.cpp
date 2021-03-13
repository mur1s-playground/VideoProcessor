#include "CameraControl.h"

#include "ApplicationGraph.h"
#include "MainUI.h"

#include <fstream>

#include "Vector2.h"

#include "Logger.h"

void camera_control_write_control(int id, struct vector2<int> top_left, struct vector2<int> bottom_right, struct cam_sensors cs);

void camera_control_init(struct camera_control* cc, int camera_count, string camera_meta_path, string sensors_path) {
	cc->camera_count = camera_count;

	cc->cam_meta = (struct cam_meta_data*)malloc(sizeof(struct cam_meta_data)*camera_count);
	cc->camera_meta_path = camera_meta_path;
	//TODO: read real meta data
	for (int c = 0; c < cc->camera_count; c++) {
		cc->cam_meta[c].resolution = { 640, 360 };
	}

	//TODO: get better weights
	float np_w[25] = { 0.0770159,0.0767262,0.0763825,0.0759683,0.0754595,0.0748203,0.0739941,0.0728876,0.0713356,0.0690222,0.065287,0.058668,0.0466034,0.0303119,0.0182474,0.0116283,0.00789316,0.00557978,0.00402777,0.00292125,0.00209508,0.00145584,0.000947069,0.000532801,0.000189098 };

	cc->cam_awareness = new struct cam_awareness[cc->camera_count];
	for (int c = 0; c < cc->camera_count; c++) {
		statistic_angle_denoiser_init(&cc->cam_awareness[c].north_pole, 25);
		statistic_angle_denoiser_set_weights(&cc->cam_awareness[c].north_pole, (float *)&np_w);
		
		statistic_angle_denoiser_init(&cc->cam_awareness[c].horizon, 25);
		statistic_angle_denoiser_set_weights(&cc->cam_awareness[c].horizon, (float *)&np_w);

		cc->cam_awareness[c].lens_distortion_quantization_size = { cc->cam_meta[c].resolution[0] / 16, cc->cam_meta[c].resolution[1] / 9 };
		cc->cam_awareness[c].lens_distortion_factor = (float*)malloc(cc->cam_awareness[c].lens_distortion_quantization_size[0] * cc->cam_awareness[c].lens_distortion_quantization_size[1] * sizeof(float));
	}
	
	cc->sensors_path = sensors_path;
	cc->cam_sens = nullptr;
	cc->cam_sens_timestamps = nullptr;
	
	cc->calibration = false;

	cc->vs_cams = nullptr;
	cc->smb_det = nullptr;
}


void camera_control_on_input_connect(struct application_graph_node* agn, int input_id) {
	struct camera_control* cc = (struct camera_control*)agn->component;

	if (input_id == 0) {
		cc->cam_sens = (struct cam_sensors*) malloc(cc->vs_cams->smb->slots * cc->camera_count * sizeof(struct cam_sensors));
		cc->cam_sens_timestamps = (unsigned long long*) malloc(cc->vs_cams->smb->slots * sizeof(unsigned long long));
	}
}

DWORD* camera_control_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct camera_control* cc = (struct camera_control*)agn->component;

	int saved_detections_count_total = cc->smb_det->size / (sizeof(struct vector2<int>) * 2);

	int last_cameras_frame = -1;
	int last_detection_frame = -1;

	int linelength_sensors = 40;

	while (agn->process_run) {
		application_graph_tps_balancer_timer_start(agn);

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
		int current_detection_frame = cc->smb_det->p_buf_c[cc->smb_det->slots * saved_detections_count_total * (sizeof(struct vector2<int>) * 2) + ((cc->smb_det->slots + 1) * 2)];
		shared_memory_buffer_release_r(cc->smb_det, cc->smb_det->slots);

		if (current_detection_frame != last_detection_frame) {
			if (current_detection_frame % 5 == 0) {

				shared_memory_buffer_try_r(cc->smb_det, current_detection_frame, true, 8);
				unsigned char* start_pos = &cc->smb_det->p_buf_c[current_detection_frame * saved_detections_count_total * (sizeof(struct vector2<int>) * 2)];
				struct vector2<int>* sp = (struct vector2<int>*) start_pos;
				unsigned long long detections_time = shared_memory_buffer_get_time(cc->smb_det, current_detection_frame);

				int lcf = last_cameras_frame;
				int break_cond = 10;
				/*
				while (cc->cam_sens_timestamps[lcf] > detections_time && break_cond > 0) {
					lcf--;
					break_cond--;
					if (lcf < 0) lcf += cc->vs_cams->smb->slots;
				}
				*/
				if (break_cond > 0) {
					/*
				logger("current frame");
				logger(current_detection_frame);
				*/

					for (int sd = 0; sd < saved_detections_count_total; sd++) {
						if (sp[0][0] == -1 && sp[0][1] == -1 && sp[1][0] == -1 && sp[1][1] == -1) break;
						int center_x = sp[1][1] - 0.5 * (sp[1][1] - sp[0][1]);
						int center_y = sp[1][0] - 0.5 * (sp[1][0] - sp[0][0]);
						int cam_id = center_y / 360;
						struct cam_sensors current_sensors;
						current_sensors.compass = 0;
						//memcpy(&current_sensors, &cc->cam_sens[lcf * cc->camera_count + cam_id], sizeof(struct cam_sensors));
						//camera_control_write_control(cam_id, sp[0], sp[1], current_sensors);
						/*
						logger("det");
						logger(sd);
						logger(sp[0][0]);
						logger(sp[0][1]);
						logger(sp[1][0]);
						logger(sp[1][1]);
						*/
						sp += 2;
					}
					//logger("current_frame_end");
				}



				shared_memory_buffer_release_r(cc->smb_det, current_detection_frame);
				if (cc->calibration) {

				}
			}
			last_detection_frame = current_detection_frame;
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
	
	camera_control_init(cc, camera_count, m_path, s_path);
}

void camera_control_destroy(struct application_graph_node* agn) {

}

int camera_control_written = 0;
void camera_control_write_control(int id, struct vector2<int> top_left, struct vector2<int> bottom_right, struct cam_sensors cs) {
	int center_x = bottom_right[1] - 0.5 * (bottom_right[1] - top_left[1]);
	int center_y = bottom_right[0] - 0.5 * (bottom_right[0] - top_left[0]);

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