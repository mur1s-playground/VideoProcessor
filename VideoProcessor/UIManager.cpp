#include "UIManager.h"

#include "GPUComposerUI.h"
#include "GPUComposerElementUI.h"
#include "GPUDenoiseUI.h"
#include "GPUMemoryBufferUI.h"
#include "GPUVideoAlphaMergeUI.h"
#include "ImShowUI.h"
#include "MaskRCNNUI.h"
#include "SharedMemoryBufferUI.h"
#include "VideoSourceUI.h"
#include "GPUMotionBlurUI.h"
#include "GPUGaussianBlurUI.h"
#include "ApplicationGraphNodeSettingsUI.h"
#include "GPUEdgeFilterUI.h"
#include "GPUPaletteFilterUI.h"
#include "GPUAudioVisualUI.h"
#include "AudioSourceUI.h"
#include "MiniGineUI.h"
#include "GPUGreenScreenUI.h"

#include "MainUI.h"

vector<pair<enum application_graph_component_type, void*>> ui_manager_frame_store;

void ui_manager_show_frame(enum application_graph_component_type agct, int node_graph_id, int node_id) {
	for (int i = 0; i < ui_manager_frame_store.size(); i++) {
		enum application_graph_component_type agct_s = ui_manager_frame_store[i].first;
		if (agct_s == agct) {
			switch (agct_s) {
			case AGCT_ANY_NODE_SETTINGS: {
				ApplicationGraphNodeSettingsFrame* agnf = (ApplicationGraphNodeSettingsFrame *)ui_manager_frame_store[i].second;
				if (!agnf->IsShownOnScreen()) {
					agnf->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_GPU_COMPOSER: {
				GPUComposerFrame* gcf = (GPUComposerFrame*)ui_manager_frame_store[i].second;
				if (!gcf->IsShownOnScreen()) {
					gcf->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_GPU_COMPOSER_ELEMENT: {
				GPUComposerElementFrame* gcef = (GPUComposerElementFrame*)ui_manager_frame_store[i].second;
				if (!gcef->IsShownOnScreen()) {
					gcef->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_GPU_DENOISE: {
				GPUDenoiseFrame* gdf = (GPUDenoiseFrame*)ui_manager_frame_store[i].second;
				if (!gdf->IsShownOnScreen()) {
					gdf->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_GPU_MEMORY_BUFFER: {
				GPUMemoryBufferFrame* gmbf = (GPUMemoryBufferFrame*)ui_manager_frame_store[i].second;
				if (!gmbf->IsShownOnScreen()) {
					gmbf->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_GPU_VIDEO_ALPHA_MERGE: {
				GPUVideoAlphaMergeFrame* vam = (GPUVideoAlphaMergeFrame*)ui_manager_frame_store[i].second;
				if (!vam->IsShownOnScreen()) {
					vam->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_IM_SHOW: {
				ImShowFrame* is = (ImShowFrame*)ui_manager_frame_store[i].second;
				if (!is->IsShownOnScreen()) {
					is->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_MASK_RCNN: {
				MaskRCNNFrame* mrcnnf = (MaskRCNNFrame*)ui_manager_frame_store[i].second;
				if (!mrcnnf->IsShownOnScreen()) {
					mrcnnf->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_SHARED_MEMORY_BUFFER: {
				SharedMemoryBufferFrame* smbf = (SharedMemoryBufferFrame*)ui_manager_frame_store[i].second;
				if (!smbf->IsShownOnScreen()) {
					smbf->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_VIDEO_SOURCE: {
				VideoSourceFrame* vis = (VideoSourceFrame*)ui_manager_frame_store[i].second;
				if (!vis->IsShownOnScreen()) {
					vis->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_GPU_MOTION_BLUR: {
				GPUMotionBlurFrame* mbf = (GPUMotionBlurFrame*)ui_manager_frame_store[i].second;
				if (!mbf->IsShownOnScreen()) {
					mbf->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_GPU_GAUSSIAN_BLUR: {
				GPUGaussianBlurFrame* mbf = (GPUGaussianBlurFrame*)ui_manager_frame_store[i].second;
				if (!mbf->IsShownOnScreen()) {
					mbf->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_GPU_EDGE_FILTER: {
				GPUEdgeFilterFrame* mbf = (GPUEdgeFilterFrame*)ui_manager_frame_store[i].second;
				if (!mbf->IsShownOnScreen()) {
					mbf->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_GPU_PALETTE_FILTER: {
				GPUPaletteFilterFrame* mbf = (GPUPaletteFilterFrame*)ui_manager_frame_store[i].second;
				if (!mbf->IsShownOnScreen()) {
					mbf->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_GPU_AUDIOVISUAL: {
				GPUAudioVisualFrame* mbf = (GPUAudioVisualFrame*)ui_manager_frame_store[i].second;
				if (!mbf->IsShownOnScreen()) {
					mbf->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_GPU_GREEN_SCREEN: {
				GPUGreenScreenFrame* mbf = (GPUGreenScreenFrame*)ui_manager_frame_store[i].second;
				if (!mbf->IsShownOnScreen()) {
					mbf->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_AUDIO_SOURCE: {
				AudioSourceFrame* mbf = (AudioSourceFrame*)ui_manager_frame_store[i].second;
				if (!mbf->IsShownOnScreen()) {
					mbf->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			case AGCT_MINI_GINE: {
				MiniGineFrame* mg = (MiniGineFrame*)ui_manager_frame_store[i].second;
				if (!mg->IsShownOnScreen()) {
					mg->Show(node_graph_id, node_id);
					return;
				}
				break;
			}
			default:

				break;
			}
		}
	}

	switch (agct) {
	case AGCT_ANY_NODE_SETTINGS: {
			ApplicationGraphNodeSettingsFrame* agnf = new ApplicationGraphNodeSettingsFrame((wxWindow*)myApp->frame);
			ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_ANY_NODE_SETTINGS, (void*)agnf));
			agnf->Show(node_graph_id, node_id);
			return;
			break;
		}
	case AGCT_GPU_COMPOSER: {
			GPUComposerFrame* gcf = new GPUComposerFrame((wxWindow*)myApp->frame);
			ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_GPU_COMPOSER, (void*)gcf));
			gcf->Show(node_graph_id, node_id);
			return;
			break;
		}
	case AGCT_GPU_COMPOSER_ELEMENT: {
			GPUComposerElementFrame* gcef = new GPUComposerElementFrame((wxWindow*)myApp->frame);
			ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_GPU_COMPOSER_ELEMENT, (void*)gcef));
			gcef->Show(node_graph_id, node_id);
			return;
			break;
		}
	case AGCT_GPU_DENOISE: {
			GPUDenoiseFrame* gdf = new GPUDenoiseFrame((wxWindow*)myApp->frame);
			ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_GPU_DENOISE, (void*)gdf));
			gdf->Show(node_graph_id, node_id);
			return;
			break;
		}
	case AGCT_GPU_MEMORY_BUFFER: {
			GPUMemoryBufferFrame* gmbf = new GPUMemoryBufferFrame((wxWindow*)myApp->frame);
			ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_GPU_MEMORY_BUFFER, (void*)gmbf));
			gmbf->Show(node_graph_id, node_id);
			return;
			break;
		}
	case AGCT_GPU_VIDEO_ALPHA_MERGE: {
			GPUVideoAlphaMergeFrame* vam = new GPUVideoAlphaMergeFrame((wxWindow*)myApp->frame);
			ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_GPU_VIDEO_ALPHA_MERGE, (void*)vam));
			vam->Show(node_graph_id, node_id);
			return;
			break;
		}
	case AGCT_IM_SHOW: {
			ImShowFrame* isf = new ImShowFrame((wxWindow*)myApp->frame);
			ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_IM_SHOW, (void*)isf));
			isf->Show(node_graph_id, node_id);
			return;
			break;
		}
	case AGCT_MASK_RCNN: {
			MaskRCNNFrame* mrcnnf = new MaskRCNNFrame((wxWindow*)myApp->frame);
			ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_MASK_RCNN, (void*)mrcnnf));
			mrcnnf->Show(node_graph_id, node_id);
			return;
			break;
		}
	case AGCT_SHARED_MEMORY_BUFFER: {
			SharedMemoryBufferFrame* smbf = new SharedMemoryBufferFrame((wxWindow*)myApp->frame);
			ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_SHARED_MEMORY_BUFFER, (void*)smbf));
			smbf->Show(node_graph_id, node_id);
			return;
			break;
		}
	case AGCT_VIDEO_SOURCE: {
			VideoSourceFrame* vsf = new VideoSourceFrame((wxWindow*)myApp->frame);
			ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)vsf));
			vsf->Show(node_graph_id, node_id);
			return;
			break;
		}
	case AGCT_GPU_MOTION_BLUR: {
			GPUMotionBlurFrame* mbf = new GPUMotionBlurFrame((wxWindow*)myApp->frame);
			ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_GPU_MOTION_BLUR, (void*)mbf));
			mbf->Show(node_graph_id, node_id);
			return;
			break;
		}
	case AGCT_GPU_GAUSSIAN_BLUR: {
			GPUGaussianBlurFrame* mbf = new GPUGaussianBlurFrame((wxWindow*)myApp->frame);
			ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_GPU_GAUSSIAN_BLUR, (void*)mbf));
			mbf->Show(node_graph_id, node_id);
			return;
			break;
		}
	case AGCT_GPU_EDGE_FILTER: {
			GPUEdgeFilterFrame* mbf = new GPUEdgeFilterFrame((wxWindow*)myApp->frame);
			ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_GPU_EDGE_FILTER, (void*)mbf));
			mbf->Show(node_graph_id, node_id);
			return;
			break;
		}
	case AGCT_GPU_PALETTE_FILTER: {
			GPUPaletteFilterFrame* mbf = new GPUPaletteFilterFrame((wxWindow*)myApp->frame);
			ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_GPU_PALETTE_FILTER, (void*)mbf));
			mbf->Show(node_graph_id, node_id);
			return;
			break;
	}
	case AGCT_GPU_AUDIOVISUAL: {
		GPUAudioVisualFrame* mbf = new GPUAudioVisualFrame((wxWindow*)myApp->frame);
		ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_GPU_AUDIOVISUAL, (void*)mbf));
		mbf->Show(node_graph_id, node_id);
		return;
		break;
	}
	case AGCT_GPU_GREEN_SCREEN: {
		GPUGreenScreenFrame* mbf = new GPUGreenScreenFrame((wxWindow*)myApp->frame);
		ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_GPU_GREEN_SCREEN, (void*)mbf));
		mbf->Show(node_graph_id, node_id);
		return;
		break;
	}
	case AGCT_AUDIO_SOURCE: {
		AudioSourceFrame* mbf = new AudioSourceFrame((wxWindow*)myApp->frame);
		ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_AUDIO_SOURCE, (void*)mbf));
		mbf->Show(node_graph_id, node_id);
		return;
		break;
	}
	case AGCT_MINI_GINE: {
		MiniGineFrame* mg = new MiniGineFrame((wxWindow*)myApp->frame);
		ui_manager_frame_store.push_back(pair<enum application_graph_component_type, void*>(AGCT_MINI_GINE, (void*)mg));
		mg->Show(node_graph_id, node_id);
		return;
		break;
	}
	default:
		break;
	}
}