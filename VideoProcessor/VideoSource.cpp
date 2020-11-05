#include "VideoSource.h"

#include "cuda_runtime.h"
#include <string>

#include "ApplicationGraph.h"
#include "MainUI.h"

#include "CUDAStreamHandler.h"

#include <sstream>
#include <fstream>

#include "Logger.h"

void video_source_set_meta(struct video_source* vs) {
	if (vs->is_open) {
		vs->video_width = vs->video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
		vs->video_height = vs->video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
		vs->video_channels = 3;
		vs->smb_size_req = vs->video_width * vs->video_height * vs->video_channels;
	}
	vs->smb = nullptr;
	vs->gmb = nullptr;
	vs->mats = nullptr;
}

void video_source_init(struct video_source* vs, int device_id) {
	stringstream ss_name;
	ss_name << device_id;
	vs->name = ss_name.str();

	vs->video_capture.open(device_id);
	vs->is_open = vs->video_capture.isOpened();
	video_source_set_meta(vs);
}

void video_source_init(struct video_source* vs, const char* path) {
	stringstream ss_name;
	ss_name << path;
	vs->name = ss_name.str();

	const char* str = vs->name.c_str();
	vs->read_hwnd = false;
	
	if (strstr(str, "dummy") == str) {
		vs->read_video_capture = false;
	} else {
		if (strstr(str, "desktop") == str) {
			vs->hwnd_desktop = GetDesktopWindow();
			vs->read_hwnd = true;
			vs->read_video_capture = false;
		} else {
			vs->read_video_capture = true;
			vs->video_capture.open(path);
			vs->is_open = vs->video_capture.isOpened();
			video_source_set_meta(vs);
		}
	}
}

void video_source_on_input_connect(struct application_graph_node *agn, int input_id) {
	if (input_id == 0) {
		struct video_source* vs = (struct video_source*)agn->component;

		if (vs->mats == nullptr) {
			vs->mats = new Mat[vs->smb->slots];
			for (int i = 0; i < vs->smb->slots; i++) {
				if (vs->video_channels == 1) {
					vs->mats[i] = Mat(vs->video_height, vs->video_width, CV_8UC1, &vs->smb->p_buf_c[i * vs->video_channels * vs->video_height * vs->video_width]);
				}
				else if (vs->video_channels == 3) {
					vs->mats[i] = Mat(vs->video_height, vs->video_width, CV_8UC3, &vs->smb->p_buf_c[i * vs->video_channels * vs->video_height * vs->video_width]);
				}
				else if (vs->video_channels == 4) {
					vs->mats[i] = Mat(vs->video_height, vs->video_width, CV_8UC4, &vs->smb->p_buf_c[i * vs->video_channels * vs->video_height * vs->video_width]);
				}
			}
			vs->smb_last_used_id = vs->smb->slots - 1;
			vs->smb_framecount = vs->smb->slots;
		}
	}
}

DWORD* video_source_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct video_source* vs = (struct video_source*)agn->component;

	if (vs->smb == nullptr) return NULL;

	if (vs->direction_smb_to_gmb) {
			int last_id = -1;
			while (agn->process_run) {
				application_graph_tps_balancer_timer_start(agn);
				int next_id = -1;
				if (vs->read_video_capture) {
					next_id = (vs->smb_last_used_id + 1) % vs->smb_framecount;
					shared_memory_buffer_try_rw(vs->smb, next_id, true, 8);
					vs->video_capture >> vs->mats[next_id];
					shared_memory_buffer_set_time(vs->smb, next_id, application_graph_tps_balancer_get_time());
					shared_memory_buffer_release_rw(vs->smb, next_id);
					if (vs->mats[next_id].empty()) {
						vs->is_open = false;
						agn->process_run = false;
						break;
					}
				} else if (vs->read_hwnd) {
					next_id = (vs->smb_last_used_id + 1) % vs->smb_framecount;
					shared_memory_buffer_try_rw(vs->smb, next_id, true, 8);
					HDC hwindowDC, hwindowCompatibleDC;
					int height, width, srcheight, srcwidth;
					HBITMAP hbwindow;
					BITMAPINFOHEADER  bi;
					hwindowDC = GetDC(vs->hwnd_desktop);
					hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
					SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);
					RECT windowsize;    // get the height and width of the screen
					GetClientRect(vs->hwnd_desktop, &windowsize);

					srcheight = windowsize.bottom;
					srcwidth = windowsize.right;
					height = vs->video_height;  //change this to whatever size you want to resize to
					width = vs->video_width;

					hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
					bi.biSize = sizeof(BITMAPINFOHEADER);    //http://msdn.microsoft.com/en-us/library/windows/window/dd183402%28v=vs.85%29.aspx
					bi.biWidth = width;
					bi.biHeight = -height;  //this is the line that makes it draw upside down or not
					bi.biPlanes = 1;
					bi.biBitCount = 32;
					bi.biCompression = BI_RGB;
					bi.biSizeImage = 0;
					bi.biXPelsPerMeter = 0;
					bi.biYPelsPerMeter = 0;
					bi.biClrUsed = 0;
					bi.biClrImportant = 0;

					// use the previously created device context with the bitmap
					SelectObject(hwindowCompatibleDC, hbwindow);
					// copy from the window device context to the bitmap device context
					StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, 0, 0, srcwidth, srcheight, SRCCOPY); //change SRCCOPY to NOTSRCCOPY for wacky colors !
					GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, vs->mats[next_id].data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);  //copy from hwindowCompatibleDC to hbwindow

					// avoid memory leak
					DeleteObject(hbwindow);
					DeleteDC(hwindowCompatibleDC);
					ReleaseDC(vs->hwnd_desktop, hwindowDC);

					shared_memory_buffer_set_time(vs->smb, next_id, application_graph_tps_balancer_get_time());
					shared_memory_buffer_release_rw(vs->smb, next_id);
				} else {
					shared_memory_buffer_try_r(vs->smb, vs->smb_framecount, true, 8);
					next_id = vs->smb->p_buf_c[vs->smb_framecount * vs->video_channels * vs->video_height * vs->video_width + ((vs->smb_framecount + 1) * 2)];
					shared_memory_buffer_release_r(vs->smb, vs->smb_framecount);
				}
				if (last_id != next_id) {
					if (vs->do_copy) {
						gpu_memory_buffer_try_r(vs->gmb, vs->gmb->slots, true, 8);
						int next_gpu_id = (vs->gmb->p_rw[2 * (vs->gmb->slots + 1)] + 1) % vs->gmb->slots;
						gpu_memory_buffer_release_r(vs->gmb, vs->gmb->slots);
						gpu_memory_buffer_try_rw(vs->gmb, next_gpu_id, true, 8);
						shared_memory_buffer_try_r(vs->smb, next_id, true, 8);
						cudaMemcpyAsync(vs->gmb->p_device + (next_gpu_id * vs->video_channels * vs->video_height * vs->video_width), &vs->smb->p_buf_c[next_id * vs->video_channels * vs->video_height * vs->video_width], vs->video_channels * vs->video_height * vs->video_width, cudaMemcpyHostToDevice, cuda_streams[0]);
						cudaStreamSynchronize(cuda_streams[0]);
						gpu_memory_buffer_set_time(vs->gmb, next_gpu_id, shared_memory_buffer_get_time(vs->smb, next_id));
						shared_memory_buffer_release_r(vs->smb, next_id);
						gpu_memory_buffer_release_rw(vs->gmb, next_gpu_id);
						gpu_memory_buffer_try_rw(vs->gmb, vs->gmb->slots, true, 8);
						vs->gmb->p_rw[2 * (vs->gmb->slots + 1)] = next_gpu_id;
						gpu_memory_buffer_release_rw(vs->gmb, vs->gmb->slots);
					}
					shared_memory_buffer_try_rw(vs->smb, vs->smb_framecount, true, 8);
					//slots													   //rw-locks									   //meta
					vs->smb->p_buf_c[vs->smb_framecount * vs->video_channels * vs->video_height * vs->video_width + ((vs->smb_framecount + 1) * 2)] = next_id;
					shared_memory_buffer_release_rw(vs->smb, vs->smb_framecount);
					last_id = next_id;
					vs->smb_last_used_id = next_id;
				}
				application_graph_tps_balancer_timer_stop(agn);
				application_graph_tps_balancer_sleep(agn);
			}
	} else if (!vs->read_video_capture && vs->do_copy && !vs->direction_smb_to_gmb) {
		int last_gpu_id = -1;
		while (agn->process_run) {
			application_graph_tps_balancer_timer_start(agn);
			gpu_memory_buffer_try_r(vs->gmb, vs->gmb->slots, true, 8);
			int next_gpu_id = vs->gmb->p_rw[2 * (vs->gmb->slots + 1)];
			gpu_memory_buffer_release_r(vs->gmb, vs->gmb->slots);
			if (next_gpu_id != last_gpu_id) {
				gpu_memory_buffer_try_r(vs->gmb, next_gpu_id, true, 8);
				int next_id = (vs->smb_last_used_id + 1) % vs->smb_framecount;
				shared_memory_buffer_try_rw(vs->smb, next_id, true, 8);
				cudaMemcpyAsync(&vs->smb->p_buf_c[next_id * vs->video_channels * vs->video_height * vs->video_width], vs->gmb->p_device + (next_gpu_id * vs->video_channels * vs->video_height * vs->video_width), vs->video_channels * vs->video_height * vs->video_width, cudaMemcpyDeviceToHost, cuda_streams[4]);
				cudaStreamSynchronize(cuda_streams[4]);
				shared_memory_buffer_set_time(vs->smb, next_id, gpu_memory_buffer_get_time(vs->gmb, next_gpu_id));
				shared_memory_buffer_release_rw(vs->smb, next_id);
				gpu_memory_buffer_release_r(vs->gmb, next_gpu_id);
				shared_memory_buffer_try_rw(vs->smb, vs->smb_framecount, true, 8);
				vs->smb->p_buf_c[vs->smb_framecount * vs->video_channels * vs->video_height * vs->video_width + ((vs->smb_framecount + 1) * 2)] = next_id;
				shared_memory_buffer_release_rw(vs->smb, vs->smb_framecount);
				vs->smb_last_used_id = next_id;
				last_gpu_id = next_gpu_id;
			}
			application_graph_tps_balancer_timer_stop(agn);
			application_graph_tps_balancer_sleep(agn);
		}
	}
	
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void video_source_externalise(struct application_graph_node* agn, string& out_str) {
	struct video_source* vs = (struct video_source*)agn->component;

	stringstream s_out;
	s_out << vs->name << std::endl;
	s_out << vs->video_width << std::endl;
	s_out << vs->video_height << std::endl;
	s_out << vs->video_channels << std::endl;
	s_out << vs->read_video_capture << std::endl;
	s_out << vs->read_hwnd << std::endl;
	s_out << vs->do_copy << std::endl;
	s_out << vs->direction_smb_to_gmb << std::endl;
	
	out_str = s_out.str();
}

void video_source_load(struct video_source* vs, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	video_source_init(vs, line.c_str());
	
	std::getline(in_f, line);
	vs->video_width = stoi(line);
	std::getline(in_f, line);
	vs->video_height = stoi(line);
	std::getline(in_f, line);
	vs->video_channels = stoi(line);
	std::getline(in_f, line);
	vs->read_video_capture = stoi(line) == 1;
	std::getline(in_f, line);
	vs->read_hwnd = stoi(line) == 1;
	std::getline(in_f, line);
	vs->do_copy = stoi(line) == 1;
	std::getline(in_f, line);
	vs->direction_smb_to_gmb = stoi(line) == 1;
}

//TODO: complete
void video_source_destory(struct application_graph_node* agn) {
	struct video_source* vs = (struct video_source*)agn->component;
	if (vs->mats != nullptr) {
		delete vs->mats;
	}
	delete vs;
}