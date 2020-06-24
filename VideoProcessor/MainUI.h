#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "VideoSourceUI.h"
#include "SharedMemoryBufferUI.h"
#include "ImShowUI.h"
#include "MaskRCNNUI.h"
#include "GPUVideoAlphaMergeUI.h"
#include "GPUMemoryBufferUI.h"
#include "GPUDenoiseUI.h"
#include "GPUComposerUI.h"
#include "GPUComposerElementUI.h"

class BasicDrawPane : public wxPanel {
    enum MenuIDs { MENU_ID_SHARED_MEMORY_BUFFER = wxID_HIGHEST + 1, MENU_ID_VIDEO_SOURCE, MENU_ID_IM_SHOW, MENU_ID_MASK_RCNN, MENU_ID_GPU_VIDEO_ALPHA_MERGE, MENU_ID_GPU_MEMORY_BUFFER, MENU_ID_GPU_DENOISE, MENU_ID_GPU_COMPOSER, MENU_ID_GPU_COMPOSER_ELEMENT, MENU_ID_GPU_MOTION_BLUR, MENU_ID_GPU_GAUSSIAN_BLUR, MENU_ID_EDIT_NODE, MENU_ID_EDIT_NODE_SETTINGS, MENU_ID_DELETE_NODE, MENU_ID_START_NODE };

public:
    int mouse_position_x, mouse_position_y;
    int mouse_down_mouse_x, mouse_down_mouse_y;
    int right_click_mouse_x, right_click_mouse_y;
    
    BasicDrawPane(wxFrame* parent);

    void paintEvent(wxPaintEvent& evt);
    void paintNow();

    void render(wxDC& dc);

    // some useful events
    void mouseMoved(wxMouseEvent& event);
    void mouseDown(wxMouseEvent& event);
    /*
    void mouseWheelMoved(wxMouseEvent& event);
    */
    void mouseReleased(wxMouseEvent& event);
    
    void rightClick(wxMouseEvent& event);
    
    //void mouseLeftWindow(wxMouseEvent& event);
    void keyPressed(wxKeyEvent& event);
    void keyReleased(wxKeyEvent& event);

    void OnShowContextMenu(wxMouseEvent& event);
    void OnContextMenuSelected(wxCommandEvent& event);

    DECLARE_EVENT_TABLE()
};

class MainFrame : public wxFrame {
    enum MenuBarIDs { MENUBAR_ID_OPEN = wxID_HIGHEST + 1, MENUBAR_ID_SAVE, MENUBAR_ID_QUIT };

public:
    MainFrame(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = 541072960L, const wxString& name = wxFrameNameStr);

    wxMenuBar* m_pMenuBar;
    wxMenu* m_pFileMenu;

    void OnMenuBarSelected(wxCommandEvent& event);
};


class MyApp : public wxApp {
    bool OnInit();
  
public:
    MainFrame* frame;
    BasicDrawPane* drawPane;
};

extern MyApp *myApp;