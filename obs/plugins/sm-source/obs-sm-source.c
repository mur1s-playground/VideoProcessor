#include <obs-module.h>
#include <util/platform.h>
#include <graphics/image-file.h>

#define WIN32_LEAN_AND_MEAN

#include <windows.h>

struct sm_source {
	obs_source_t *source;

	char *sm_name;
	int sm_width;
	int sm_height;
	int sm_slots;
	int fps_target;
	bool visible;

	char *sm_cname;
	HANDLE h_mutex;
	HANDLE h_map_file;
	LPCTSTR buffer;
	int last_frame_id;
	bool initialised;

	gs_image_file2_t if2;
	bool texture_loaded;

	float timer;
};

static const char *sm_source_getname(void *unused) {
	UNUSED_PARAMETER(unused);
	return "Shared Memory Source";
}

static uint32_t sm_source_getheight(void *data) {
	struct sm_source *s = data;
	return s->sm_height;
}

static uint32_t sm_source_getwidth(void *data) {
	struct sm_source *s = data;
	return s->sm_width;
}

static void sm_source_show(void *data) {
	struct sm_source *s = data;
	s->visible = true;
}

static void sm_source_hide(void *data) {
	struct sm_source *s = data;
	s->visible = false;
}

static void sm_source_unload(struct sm_source *s) {
	s->visible = false;
	if (s->texture_loaded) {
		s->texture_loaded = false;
		obs_enter_graphics();
		gs_image_file2_free(&s->if2);
		obs_leave_graphics();
	}

	if (s->initialised) {
		s->initialised = false;
		if (s->h_mutex != NULL) {
			CloseHandle(s->h_mutex);
		}
		if (s->h_map_file != NULL) {
			CloseHandle(s->h_map_file);
		}
	}
}

static void sm_source_load(struct sm_source *s) {
	obs_enter_graphics();
	gs_image_file2_free(&s->if2);
	obs_leave_graphics();

	if (s->initialised) {
		blog(0, "[sm_source_]: init texture");

		memset(&s->if2.image, 0, sizeof(s->if2.image));

		s->if2.image.texture_data =
			bmalloc(s->sm_width * s->sm_height * 4);
		s->if2.image.format = GS_BGRA;
		s->if2.image.cx = s->sm_width;
		s->if2.image.cy = s->sm_height;

		s->if2.mem_usage = s->sm_width * s->sm_height * 4;
		s->if2.image.is_animated_sm = true;

		s->if2.image.loaded = !!s->if2.image.texture_data;

		obs_enter_graphics();
		gs_image_file2_init_texture(&s->if2);
		obs_leave_graphics();

		if (!s->if2.image.loaded) {
			blog(0, "[sm_source_]: failed to load texture");
		} else {
			s->texture_loaded = true;
			if (obs_source_showing(s->source)) {
				sm_source_show(s);
			}
		}
	}
}

static void sm_source_update(void *data, obs_data_t *settings) {
	struct sm_source *s = data;

	s->initialised = false;
	s->last_frame_id = -1;
	s->sm_width = (int)obs_data_get_int(settings, "sm_width");
	s->sm_height = (int)obs_data_get_int(settings, "sm_height");
	s->sm_slots = (int)obs_data_get_int(settings, "sm_slots");
	s->fps_target = (int)obs_data_get_int(settings, "fps_target");
	const char *name = obs_data_get_string(settings, "sm_name");
	s->sm_name = bmalloc(strlen(name) + 1);
	snprintf(s->sm_name, strlen(name) + 1, "%s", name);
	s->texture_loaded = false;
	s->visible = false;

	sm_source_unload(s);

	TCHAR szName[256];
	TCHAR szNameM[256];

	int wchars_num = MultiByteToWideChar(CP_UTF8, 0, s->sm_name,
					     -1, NULL, 0);
	MultiByteToWideChar(CP_UTF8, 0, s->sm_name, -1, szName,
			    wchars_num);

	char tmp[256];
	snprintf(tmp, 256, "c%s", s->sm_name);

	wchars_num++;

	MultiByteToWideChar(CP_UTF8, 0, tmp, -1, szNameM, wchars_num);

	s->h_mutex = CreateMutex(NULL, FALSE, szNameM);
	if (s->h_mutex != NULL) {
		blog(0, "[sm_source_]: h_mutex == init");

		s->h_map_file =	OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, szName);
		if (s->h_map_file != NULL) {
			blog(0, "[sm_source_]: h_map_file == init");

			s->buffer = (LPTSTR)MapViewOfFile(
				s->h_map_file, FILE_MAP_ALL_ACCESS,
				      //frame data				     //rw mutex data	     //most recent frame_id	//timestamps
				0, 0, s->sm_width * s->sm_height * 4 * s->sm_slots + 2 * (s->sm_slots + 1) + sizeof(int)		+ (s->sm_slots * sizeof(unsigned long long)));
			if (s->buffer == NULL) {
				blog(0, "[sm_source_]: dynamic_buffer == null");
			} else {
				blog(0, "[sm_source_]: dynamic_buffer == init");
				s->initialised = true;
				sm_source_load(s);
			}


		} else {
			blog(0, "[sm_source_]: h_map_file == null");
			CloseHandle(s->h_mutex);
		}


	} else {
		blog(0, "[sm_source_]: h_mutex == null");
	}
}

static void *sm_source_create(obs_data_t *settings, obs_source_t *source) {
	UNUSED_PARAMETER(settings);

	struct sm_source *s = bzalloc(sizeof(struct sm_source));
	s->source = source;
	s->h_mutex = NULL;
	s->h_map_file = NULL;
	s->timer = 0.0f;

	sm_source_update(s, settings);

	return s;
}

static void sm_source_destroy(void *data) {
	struct sm_source *s = data;

	bfree(s->sm_name);

	sm_source_unload(s);
	bfree(s);
}

static void sm_source_defaults(obs_data_t *settings) {
	obs_data_set_default_int(settings, "sm_width", 1920);
	obs_data_set_default_int(settings, "sm_height", 1080);
	obs_data_set_default_int(settings, "sm_slots", 30);
	obs_data_set_default_int(settings, "fps_target", 30);
}

static obs_properties_t *sm_source_getproperties(void *data) {
	struct sm_source *s = data;
	UNUSED_PARAMETER(data);

	obs_properties_t *props = obs_properties_create();

	obs_properties_set_flags(props, OBS_PROPERTIES_DEFER_UPDATE);

	obs_property_t *prop;
	
	prop = obs_properties_add_int(props, "sm_width", "width", 0, 1920, 1);
	prop = obs_properties_add_int(props, "sm_height", "height", 0, 1920, 1);
	prop = obs_properties_add_int(props, "sm_slots", "slots", 0, 120, 1);
	prop = obs_properties_add_int(props, "fps_target", "fps target", 0, 120, 1);

	prop = obs_properties_add_text(props, "sm_name", "name", OBS_TEXT_DEFAULT);

	return props;
}

static void sm_buffer_lock_rw(struct sm_source *s, int slot) {
	unsigned char *p_buffer = (unsigned char *)s->buffer;

	WaitForSingleObject(s->h_mutex, INFINITE);
	bool is_written_to = (p_buffer[s->sm_width*s->sm_height*4 * s->sm_slots + 2 * slot + 1] == 1);
	bool is_read_from = (p_buffer[s->sm_width*s->sm_height*4 * s->sm_slots + 2 * slot] > 0);
	bool set_write = false;
	while (true) {
		if (!is_written_to) {
			p_buffer[s->sm_width * s->sm_height * 4 * s->sm_slots + 2 * slot + 1] = 1;
			set_write = true;
		}
		if (!is_read_from && set_write) {
			ReleaseMutex(s->h_mutex);
			return;
		}
		ReleaseMutex(s->h_mutex);
		Sleep(8);
		WaitForSingleObject(s->h_mutex, INFINITE);
		is_written_to = (p_buffer[s->sm_width*s->sm_height*4 * s->sm_slots + 2 * slot + 1] == 1);
		is_read_from = (p_buffer[s->sm_width*s->sm_height*4 * s->sm_slots + 2 * slot] > 0);
	}
}

static void sm_buffer_release_rw(struct sm_source *s, int slot) {
	unsigned char *p_buffer = (unsigned char *)s->buffer;

	WaitForSingleObject(s->h_mutex, INFINITE);
	p_buffer[s->sm_width * s->sm_height * 4 * s->sm_slots + 2 * slot + 1] =	0;
	ReleaseMutex(s->h_mutex);
}

static void sm_buffer_lock_r(struct sm_source *s, int slot) {
	unsigned char *p_buffer = (unsigned char *)s->buffer;

	WaitForSingleObject(s->h_mutex, INFINITE);
	bool is_written_to = (p_buffer[s->sm_width * s->sm_height * 4 * s->sm_slots + 2 * slot + 1] == 1);
	while (true) {
		if (!is_written_to) {
			p_buffer[s->sm_width * s->sm_height * 4 * s->sm_slots + 2 * slot]++;
			ReleaseMutex(s->h_mutex);
			return;
		}
		ReleaseMutex(s->h_mutex);
		Sleep(8);
		WaitForSingleObject(s->h_mutex, INFINITE);
		is_written_to = (p_buffer[s->sm_width * s->sm_height * 4 * s->sm_slots + 2 * slot + 1] == 1);
	}
}

static void sm_buffer_release_r(struct sm_source *s, int slot) {
	unsigned char *p_buffer = (unsigned char *)s->buffer;

	WaitForSingleObject(s->h_mutex, INFINITE);
	p_buffer[s->sm_width * s->sm_height * 4 * s->sm_slots + 2 * slot]--;
	ReleaseMutex(s->h_mutex);
}



static void sm_source_tick(void *data, float seconds) {
	struct sm_source *s = data;

	s->timer -= seconds;
	if (s->timer > 0.0f) return;
	s->timer += (1.0f / (float)s->fps_target);

	if (s->initialised && s->visible) {
		int frame_id = 1;
		int dim = s->sm_width * s->sm_height;

		unsigned char *p_buffer = (unsigned char *)s->buffer;

		sm_buffer_lock_r(s, s->sm_slots);
		memcpy(&frame_id, &p_buffer[s->sm_width * s->sm_height * 4 * s->sm_slots + 2 * (s->sm_slots+1)], sizeof(int));
		sm_buffer_release_r(s, s->sm_slots);

		if (frame_id >= 0 && frame_id < s->sm_slots && frame_id != s->last_frame_id) {
			sm_buffer_lock_r(s, frame_id);
			
			obs_enter_graphics();
			gs_texture_set_image(s->if2.image.texture, &p_buffer[frame_id * dim * 4], gs_texture_get_width(s->if2.image.texture) * 4, false);
			obs_leave_graphics();

			sm_buffer_release_r(s, frame_id);
			s->last_frame_id = frame_id;
		}
	}

	
}

static void sm_source_render(void *data, gs_effect_t *effect) {
	struct sm_source *s = data;

	if (!s->if2.image.texture)
		return;

	gs_effect_set_texture(gs_effect_get_param_by_name(effect, "image"),
			      s->if2.image.texture);
	gs_draw_sprite(s->if2.image.texture, 0, s->if2.image.cx,
		       s->if2.image.cy);
}

struct obs_source_info sm_source_info = {
	.id = "sm_source",
	.type = OBS_SOURCE_TYPE_INPUT,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = sm_source_getname,
	.create = sm_source_create,
	.destroy = sm_source_destroy,
	.update = sm_source_update,
	.get_defaults = sm_source_defaults,
	.show = sm_source_show,
	.hide = sm_source_hide,
	.get_width = sm_source_getwidth,
	.get_height = sm_source_getheight,
	.video_render = sm_source_render,
	.video_tick = sm_source_tick,
	.get_properties = sm_source_getproperties,
	.icon_type = OBS_ICON_TYPE_MEDIA,
};

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("sm-source", "en-US")
MODULE_EXPORT const char *obs_module_description(void)
{
	return "Shared Memory sources";
}

bool obs_module_load(void) {
	obs_register_source(&sm_source_info);
	return true;
}
