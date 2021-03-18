#include "CameraControlDiagnostic.h"
#include "CameraControlDiagnosticKernel.h"

#include "ApplicationGraph.h"
#include "MainUI.h"

#include "CUDAStreamHandler.h"

#include "Logger.h"

void camera_control_diagnostic_init(struct camera_control_diagnostic* ccd) {
	ccd->cc = nullptr;
	ccd->cc_shared_state_gpu = nullptr;
	ccd->vs_out = nullptr;
}

void camera_control_diagnostic_on_input_connect(struct application_graph_node* agn, int input_id) {
	struct camera_control_diagnostic* ccd = (struct camera_control_diagnostic*)agn->component;

	if (input_id == 0) {
		ccd->cc_shared_state_gpu = new struct gpu_memory_buffer();
		gpu_memory_buffer_init(ccd->cc_shared_state_gpu, "cc_shared_state", ccd->cc->shared_state_size_req, ccd->cc->smb_shared_state->slots, sizeof(int));
		
	}
}

DWORD* camera_control_diagnostic_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct camera_control_diagnostic* ccd = (struct camera_control_diagnostic*)agn->component;

	int next_frame_out = -1;
	int last_shared_state = -1;
	while (agn->process_run) {
		application_graph_tps_balancer_timer_start(agn);

		shared_memory_buffer_try_rw(ccd->cc->smb_shared_state, ccd->cc->smb_shared_state->slots, true, 8);
		int current_shared_state = ccd->cc->smb_shared_state->p_buf_c[ccd->cc->smb_shared_state->slots * ccd->cc->smb_shared_state->size + ((ccd->cc->smb_shared_state->slots + 1) * 2)];
		shared_memory_buffer_release_rw(ccd->cc->smb_shared_state, ccd->cc->smb_shared_state->slots);
		//logger("index", current_shared_state);
		if (current_shared_state != last_shared_state) {
			shared_memory_buffer_try_r(ccd->cc->smb_shared_state, current_shared_state, true, 8);
			gpu_memory_buffer_try_rw(ccd->cc_shared_state_gpu, current_shared_state, true, 8);
			cudaMemcpyAsync(ccd->cc_shared_state_gpu->p_device + (current_shared_state * ccd->cc->smb_shared_state->size), &ccd->cc->smb_shared_state->p_buf_c[current_shared_state * ccd->cc->smb_shared_state->size], ccd->cc->smb_shared_state->size, cudaMemcpyHostToDevice, cuda_streams[0]);
			cudaStreamSynchronize(cuda_streams[0]);
			gpu_memory_buffer_release_rw(ccd->cc_shared_state_gpu, current_shared_state);
			shared_memory_buffer_release_r(ccd->cc->smb_shared_state, current_shared_state);
			
			/* using kernel call parameter instead, because i use slot idx counter from cc
			gpu_memory_buffer_try_rw(ccd->cc_shared_state_gpu, ccd->cc_shared_state_gpu->slots, true, 8);
			ccd->cc_shared_state_gpu->p_rw[2 * (ccd->cc_shared_state_gpu->slots + 1)] = current_shared_state;
			gpu_memory_buffer_release_rw(ccd->cc_shared_state_gpu, ccd->cc_shared_state_gpu->slots);
			*/

			gpu_memory_buffer_try_r(ccd->vs_out->gmb, ccd->vs_out->gmb->slots, true, 8);
			int next_gpu_out_id = (ccd->vs_out->gmb->p_rw[2 * (ccd->vs_out->gmb->slots + 1)] + 1) % ccd->vs_out->gmb->slots;
			gpu_memory_buffer_release_r(ccd->vs_out->gmb, ccd->vs_out->gmb->slots);

			gpu_memory_buffer_try_rw(ccd->vs_out->gmb, next_gpu_out_id, true, 8);
			gpu_memory_buffer_try_r(ccd->cc_shared_state_gpu, current_shared_state, true, 8);
			camera_control_diagnostic_launch(ccd->cc_shared_state_gpu->p_device + (current_shared_state * ccd->cc->smb_shared_state->size), ccd->cc->camera_count, ccd->vs_out->gmb->p_device + (next_gpu_out_id * ccd->vs_out->gmb->size), ccd->vs_out->video_width, ccd->vs_out->video_height);
			gpu_memory_buffer_release_r(ccd->cc_shared_state_gpu, current_shared_state);
			gpu_memory_buffer_release_rw(ccd->vs_out->gmb, next_gpu_out_id);

			gpu_memory_buffer_try_rw(ccd->vs_out->gmb, ccd->vs_out->gmb->slots, true, 8);
			ccd->vs_out->gmb->p_rw[2 * (ccd->vs_out->gmb->slots + 1)] = next_gpu_out_id;
			gpu_memory_buffer_release_rw(ccd->vs_out->gmb, ccd->vs_out->gmb->slots);

			last_shared_state = current_shared_state;
		}

		application_graph_tps_balancer_timer_stop(agn);
		application_graph_tps_balancer_sleep(agn);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void camera_control_diagnostic_externalise(struct application_graph_node* agn, string& out_str) {
	struct camera_control_diagnostic* ccd = (struct camera_control_diagnostic*)agn->component;

	stringstream s_out;
	out_str = s_out.str();
}

void camera_control_diagnostic_load(struct camera_control_diagnostic* ccd, ifstream& in_f) {
	std::string line;
	
	camera_control_diagnostic_init(ccd);
}

void  camera_control_diagnostic_destroy(struct application_graph_node* agn) {
	struct camera_control_diagnostic* ccd = (struct camera_control_diagnostic*)agn->component;
	delete ccd;
}