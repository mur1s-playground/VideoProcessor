#include "ApplicationGraph.h"

#include <sstream>

#include "VideoSource.h"
#include "VideoSourceUI.h"

#include "SharedMemoryBuffer.h"
#include "SharedMemoryBufferUI.h"

#include "MaskRCNN.h"
#include "MaskRCNNUI.h"

#include "ImShow.h"
#include "ImShowUI.h"

#include "GPUComposerElement.h"
#include "GPUComposerElementUI.h"

#include "MainUI.h"

#include "Logger.h"

using namespace std;

vector <struct application_graph *>   ags;
int application_graph_hovering_node_id = -1;

int application_graph_is_on_input(int id, int node_id, int pos_x, int pos_y, float* dist_out) {
    struct application_graph_node* current_node = ags[id]->nodes[node_id];
    float closest_dist = -1.0f;
    int closest_id = -1;
    for (int in = 0; in < current_node->inputs.size(); in++) {
        int to_x = current_node->pos_x;
        int to_height_id = current_node->inputs[in].first + 1;
        int h_1 = current_node->heights[to_height_id];
        int h_2 = current_node->heights[to_height_id + 1];
        int to_y = h_1 + (h_2 - h_1) / 2;

        float dist_2 = sqrt(pow(to_x - pos_x, 2) + pow(to_y - pos_y, 2));
        if ((dist_2 < closest_dist || closest_dist < 0) && dist_2 < 6) {
            closest_dist = dist_2;
            closest_id = in;
        }
    }
    if (dist_out != nullptr) {
        *dist_out = closest_dist;
    }
    return closest_id;
}

int application_graph_is_on_output(int id, int node_id, int pos_x, int pos_y, float* dist_out) {
    struct application_graph_node* current_node = ags[id]->nodes[node_id];
    float closest_dist = -1.0f;
    int closest_id = -1;
    for (int out = 0; out < current_node->outputs.size(); out++) {
        int from_x = current_node->pos_x + 5 + current_node->max_width + 5;
        int from_height_id = current_node->outputs[out].first + 1;
        int h_1 = current_node->heights[from_height_id];
        int h_2 = current_node->heights[from_height_id + 1];
        int from_y = h_1 + (h_2 - h_1) / 2;

        float dist_1 = sqrt(pow(from_x - pos_x, 2) + pow(from_y - pos_y, 2));
        if ((dist_1 < closest_dist || closest_dist < 0) && dist_1 < 6) {
            closest_dist = dist_1;
            closest_id = out;
        }
    }
    if (dist_out != nullptr) {
        *dist_out = closest_dist;
    }
    return closest_id;
}

void application_graph_add_edge(int id, int pos_x1, int pos_y1, int pos_x2, int pos_y2) {
    int closest_nid_1 = -1;
    int closest_id_1 = -1;
    float closest_dist_1 = -1.0f;

    int closest_nid_2 = -1;
    int closest_id_2 = -1;
    float closest_dist_2 = -1.0f;
    for (int i = 0; i < ags[id]->nodes.size(); i++) {
        struct application_graph_node *current_node = ags[id]->nodes[i];
      
        float closest_dist_out = -1.0f;
        int closest_id_out = application_graph_is_on_output(id, i, pos_x1, pos_y1, &closest_dist_out);
        if (closest_id_out > -1 && (closest_dist_out < closest_dist_1 || closest_dist_1 < 0) && closest_dist_out < 6) {
            closest_dist_1 = closest_dist_out;
            closest_id_1 = closest_id_out;
            closest_nid_1 = i;
        }

        float closest_dist_in = -1.0f;
        int closest_id_in = application_graph_is_on_input(id, i, pos_x2, pos_y2, &closest_dist_in);
        if (closest_id_in > -1 && (closest_dist_in < closest_dist_2 || closest_dist_2 < 0) && closest_dist_in < 6) {
            closest_dist_2 = closest_dist_in;
            closest_id_2 = closest_id_in;
            closest_nid_2 = i;
        }
    }

    if (closest_nid_1 >= 0 && closest_nid_2 >= 0 && closest_id_1 >= 0 && closest_id_2 >= 0) {
        struct application_graph_edge *edge = new application_graph_edge();
        pair<application_graph_node*, int> from;
        pair<application_graph_node*, int> to;
        edge->from = pair<application_graph_node*, int>(ags[id]->nodes[closest_nid_1], closest_id_1);
        edge->to = pair<application_graph_node*, int>(ags[id]->nodes[closest_nid_2], closest_id_2);
        ags[id]->edges.push_back(edge);

        struct application_graph_node* current_node = ags[id]->nodes[closest_nid_2];
        struct application_graph_edge* current_edge = edge;
        
        void* input_target_ptr = current_node->inputs[closest_id_2].second.second;

        if (current_node->inputs[closest_id_2].second.first == AGCT_SHARED_MEMORY_BUFFER) {
            struct shared_memory_buffer** input_target_ptr_t = (struct shared_memory_buffer**)input_target_ptr;
            *input_target_ptr_t = (struct shared_memory_buffer*)(current_edge->from.first->outputs[current_edge->from.second].second.second);
        } else if (current_node->inputs[closest_id_2].second.first == AGCT_GPU_MEMORY_BUFFER) {
            struct gpu_memory_buffer** input_target_ptr_t = (struct gpu_memory_buffer**)input_target_ptr;
            *input_target_ptr_t = (struct gpu_memory_buffer*)(current_edge->from.first->outputs[current_edge->from.second].second.second);
        } else if (current_node->inputs[closest_id_2].second.first == AGCT_VIDEO_SOURCE) {
            struct video_source** input_target_ptr_t = (struct video_source**)input_target_ptr;
            *input_target_ptr_t = (struct video_source*)(current_edge->from.first->outputs[current_edge->from.second].second.second);
        } else if (current_node->inputs[closest_id_2].second.first == AGCT_MASK_RCNN) {
            struct mask_rcnn** input_target_ptr_t = (struct mask_rcnn**)input_target_ptr;
            *input_target_ptr_t = (struct mask_rcnn*)(current_edge->from.first->outputs[current_edge->from.second].second.second);
        } else if (current_node->inputs[closest_id_2].second.first == AGCT_GPU_COMPOSER_ELEMENT) {
            struct gpu_composer_element** input_target_ptr_t = (struct gpu_composer_element**)input_target_ptr;
            *input_target_ptr_t = (struct gpu_composer_element*)(current_edge->from.first->outputs[current_edge->from.second].second.second);
        }
   
        if (current_node->on_input_connect != nullptr) {
#ifndef FPTR_OIC_AGN_INT
#define FPTR_OIC_AGN_INT
            typedef void (*oicptr)(struct application_graph_node* agn, int);
#endif
            oicptr f_ptr = (oicptr)current_node->on_input_connect;
            f_ptr(current_node, closest_id_2);
        }
    }
}

void application_graph_start_stop_node(int id, int node_id) {
    struct application_graph_node* current_node = ags[id]->nodes[node_id];
    if (current_node->process_run) {
        application_graph_stop_node(id, node_id);
    } else {
        application_graph_start_node(id, node_id);
    }
}

void application_graph_start_node(int id, int node_id) {
    struct application_graph_node* current_node = ags[id]->nodes[node_id];

    if (current_node->process != nullptr) {
        current_node->process_run = true;
        CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)current_node->process, current_node, 0, 0);
        myApp->drawPane->Refresh();
    }
}

void application_graph_stop_node(int id, int node_id) {
    struct application_graph_node* current_node = ags[id]->nodes[node_id];
    current_node->process_run = false;
}

void application_graph_draw_nodes(struct application_graph* ag, wxDC& dc) {
    wxFont f_normal = dc.GetFont();
    wxFont f_bold = dc.GetFont();
    f_bold.MakeBold();

    for (int i = 0; i < ag->nodes.size(); i++) {
        struct application_graph_node *current_node = ag->nodes[i];

        dc.SetFont(f_bold);
        wxSize max_width = dc.GetTextExtent(current_node->name);
        max_width.SetWidth(max_width.GetWidth() + 5);
        int total_height = max_width.GetHeight();

        dc.SetFont(f_normal);
        for (int j = 0; j < current_node->v.size(); j++) {
            pair<enum application_graph_node_vtype, void*> current_v = current_node->v[j];
            stringstream ss;
            bool draw_text = true;
            if (current_v.first == AGNVT_STRING) {
                ss << *((string*)current_v.second);
            } else if (current_v.first == AGNVT_STRING_LIST) {
                draw_text = false;
                vector<string> current_string_list = *((vector<string> *) current_v.second);
                stringstream line;
                int line_c = 0;
                for (int sl = 0; sl < current_string_list.size(); sl++) {
                    if (line_c > 0) line << ",";
                    line << current_string_list[sl];
                    line_c++;
                    wxSize tmp = dc.GetTextExtent(line.str());
                    tmp.SetWidth(tmp.GetWidth() + 5);
                    if (tmp.GetWidth() > 200 || sl == current_string_list.size() - 1) {
                        total_height += tmp.GetHeight();
                        if (tmp.GetWidth() > max_width.GetWidth()) {
                            max_width = tmp;
                        }
                        line.str("");
                        line.clear();
                        line_c = 0;
                    }
                }
                
            } else if (current_v.first == AGNVT_INT) {
                int* current_int = (int*)current_v.second;
                ss << *current_int;
            } else if (current_v.first == AGNVT_FLOAT) {
                ss.precision(2);
                float* current_float = (float*)current_v.second;
                ss << *current_float;
            } else if (current_v.first == AGNVT_BOOL) {
                bool* current_bool = (bool*)current_v.second;
                if (*current_bool) {
                    ss << "true";
                } else {
                    ss << "false";
                }
            } else if (current_v.first == AGNVT_SEPARATOR) {
                draw_text = false;
                total_height += 6;
            }
            if (draw_text) {
                wxSize tmp = dc.GetTextExtent(ss.str());
                tmp.SetWidth(tmp.GetWidth() + 5);
                total_height += tmp.GetHeight();
                if (tmp.GetWidth() > max_width.GetWidth()) {
                    max_width = tmp;
                }
            }
        }

        current_node->max_width = max_width.GetWidth();

        int border_pen_size = 2;
        if (application_graph_hovering_node_id == i) {
            border_pen_size--;
        }
        if (current_node->process_run) {
            border_pen_size--;
        }
               
        dc.SetBrush(*wxWHITE_BRUSH);
        dc.SetPen(wxPen(wxColor(0, 0, 0), border_pen_size));
        dc.DrawRectangle(current_node->pos_x, current_node->pos_y, 5 + current_node->max_width + 5, total_height + 5);
        if (application_graph_hovering_node_id == i) {
            dc.SetBrush(*wxTRANSPARENT_BRUSH);
            dc.SetPen(wxPen(wxColor(255, 255, 0), 1));
            dc.DrawRectangle(current_node->pos_x, current_node->pos_y, 5 + current_node->max_width + 5, total_height + 5);
        }
        if (current_node->process_run) {
            dc.SetBrush(*wxTRANSPARENT_BRUSH);
            dc.SetPen(wxPen(wxColor(0, 255, 0), 1));
            dc.DrawRectangle(current_node->pos_x+1, current_node->pos_y+1, 5 + current_node->max_width + 3, total_height + 3);
        }
        dc.SetBrush(*wxWHITE_BRUSH);
        dc.SetPen(wxPen(wxColor(0, 0, 0), 1));

        current_node->heights.clear();

        int current_y = current_node->pos_y + 5;
        current_node->heights.push_back(current_y);

        dc.SetFont(f_bold);
        dc.DrawText(wxString(current_node->name), current_node->pos_x + 5, current_y);

        current_y += dc.GetTextExtent(current_node->name).GetHeight();
        current_node->heights.push_back(current_y);

        dc.SetFont(f_normal);
        
        for (int j = 0; j < current_node->v.size(); j++) {
            pair<enum application_graph_node_vtype, void*> current_v = current_node->v[j];
            stringstream ss;
            bool draw_text = true;
            if (current_v.first == AGNVT_STRING) {
                ss << *((string *)current_v.second);
            } else if (current_v.first == AGNVT_STRING_LIST) {
                draw_text = false;
                vector<string> current_string_list = *((vector<string>*) current_v.second);
                stringstream line;
                int line_c = 0;
                for (int sl = 0; sl < current_string_list.size(); sl++) {
                    if (line_c > 0) line << ",";
                    line << current_string_list[sl];
                    line_c++;
                    wxSize tmp = dc.GetTextExtent(line.str());
                    tmp.SetWidth(tmp.GetWidth() + 5);
                    if (tmp.GetWidth() > 200 || sl == current_string_list.size() - 1) {
                        dc.DrawText(wxString(line.str()), current_node->pos_x + 5, current_y);
                        current_y += tmp.GetHeight();
                        line.str("");
                        line.clear();
                        line_c = 0;
                    }
                }
            } else if (current_v.first == AGNVT_INT) {
                int* current_int = (int*)current_v.second;
                ss << *current_int;
            } else if (current_v.first == AGNVT_FLOAT) {
                ss.precision(2);
                float* current_float = (float*)current_v.second;
                ss << *current_float;
            } else if (current_v.first == AGNVT_BOOL) {
                bool* current_bool = (bool*)current_v.second;
                if (*current_bool) {
                    ss << "true";
                }
                else {
                    ss << "false";
                }
            } else if (current_v.first == AGNVT_SEPARATOR) {
                draw_text = false;
                dc.DrawLine(current_node->pos_x+1, current_y, current_node->pos_x + 5 + current_node->max_width + 4, current_y);
                current_y += 2;
            }
            if (draw_text) {
                string tmp = ss.str();
                dc.DrawText(wxString(tmp), current_node->pos_x + 5, current_y);
                current_y += dc.GetTextExtent(tmp).GetHeight();
            }
            current_node->heights.push_back(current_y);
        }

        dc.SetPen(wxPen(wxColor(0, 0, 0), 2));
        for (int j = 0; j < current_node->inputs.size(); j++) {
            pair<int, pair<enum application_graph_component_type, void*>> current_input = current_node->inputs[j];
            int h_1 = current_node->heights[current_input.first + 1];
            int h_2 = current_node->heights[current_input.first + 2];
            dc.DrawCircle(wxPoint(current_node->pos_x, h_1 + (h_2 - h_1)/2), 4);
        }

        for (int j = 0; j < current_node->outputs.size(); j++) {
            pair<int, pair<enum application_graph_component_type, void*>> current_output = current_node->outputs[j];
            int h_1 = current_node->heights[current_output.first + 1];
            int h_2 = current_node->heights[current_output.first + 2];
            dc.DrawCircle(wxPoint(current_node->pos_x + 5 + current_node->max_width + 3, h_1 + (h_2 - h_1) / 2), 4);
        }
    }
}

void application_graph_draw_edges(struct application_graph* ag, wxDC& dc) {
    for (int i = 0; i < ag->edges.size(); i++) {
        struct application_graph_edge* current_edge = ag->edges[i];

        int h_1, h_2;

        struct application_graph_node* from = current_edge->from.first;
        int from_id = current_edge->from.second;
        int from_x = from->pos_x + 5 + from->max_width + 5;

        int from_height_id = from->outputs[from_id].first + 1;
        h_1 = from->heights[from_height_id];
        h_2 = from->heights[from_height_id + 1];
        int from_y = h_1 + (h_2 - h_1)/2;

        struct application_graph_node *to = current_edge->to.first;
        int to_id = current_edge->to.second;
        int to_x = to->pos_x;

        int to_height_id = to->inputs[to_id].first + 1;
        h_1 = to->heights[to_height_id];
        h_2 = to->heights[to_height_id + 1];
        int to_y = h_1 + (h_2 - h_1) / 2;

        dc.DrawLine(wxPoint(from_x, from_y), wxPoint(to_x, to_y));
    }
}

void application_graph_hovering_node(int application_graph_id) {
    if (ags.size() > application_graph_id) {
        int hovering_candidate = -1;
        for (int i = 0; i < ags[application_graph_id]->nodes.size(); i++) {
            struct application_graph_node* current_node = ags[application_graph_id]->nodes[i];
            if (myApp->drawPane->mouse_position_x >= current_node->pos_x && myApp->drawPane->mouse_position_x <= current_node->pos_x + current_node->max_width &&
                myApp->drawPane->mouse_position_y >= current_node->pos_y && current_node->heights.size() > 0 && myApp->drawPane->mouse_position_y <= current_node->heights[current_node->heights.size()-1]) {
                hovering_candidate = i;
            }
        }
        if (hovering_candidate == -1 && application_graph_hovering_node_id > -1) {
            application_graph_hovering_node_id = -1;
            myApp->drawPane->Refresh();
        } else if (application_graph_hovering_node_id != hovering_candidate) {
            application_graph_hovering_node_id = hovering_candidate;
            myApp->drawPane->Refresh();
        }
    }
}