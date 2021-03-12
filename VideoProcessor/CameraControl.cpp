#include "CameraControl.h"

#include "ApplicationGraph.h"
#include "MainUI.h"

void camera_control_init(struct camera_control* cc) {
	cc->smb_det = nullptr;
}

DWORD* camera_control_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct camera_control* cc = (struct camera_control*)agn->component;

	int last_frame = -1;
	while (agn->process_run) {
		application_graph_tps_balancer_timer_start(agn);



		application_graph_tps_balancer_timer_stop(agn);
		application_graph_tps_balancer_sleep(agn);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void camera_control_externalise(struct application_graph_node* agn, string& out_str) {

}

void camera_control_load(struct camera_control* cc, ifstream& in_f) {
	cc->smb_det = nullptr;
}

void camera_control_destroy(struct application_graph_node* agn) {

}