#define CGLTF_IMPLEMENTATION
#include "../cgltf.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <vector>

static int failures = 0;

static void check_near(const char* what, double actual, double expected, double tol)
{
	if (fabs(actual - expected) > tol)
	{
		fprintf(stderr, "FAIL: %s: got %.9g, expected %.9g (tol %g)\n", what, actual, expected, tol);
		failures++;
	}
}

static void check_vec(const char* what, const cgltf_float* actual, const double* expected, cgltf_size n, double tol)
{
	for (cgltf_size i = 0; i < n; ++i)
	{
		char label[128];
		snprintf(label, sizeof(label), "%s[%u]", what, (unsigned)i);
		check_near(label, actual[i], expected[i], tol);
	}
}

/* Builds a sampler over caller-owned float arrays; all cgltf objects are stored in the
   returned struct so no glTF file is needed. */
struct TestSampler
{
	cgltf_animation_sampler sampler;
	cgltf_accessor input;
	cgltf_accessor output;
	cgltf_buffer_view input_view;
	cgltf_buffer_view output_view;
};

static void init_sampler(TestSampler* ts, const cgltf_float* times, cgltf_size keyframe_count,
	const cgltf_float* values, cgltf_type value_type, cgltf_size value_count, cgltf_interpolation_type interpolation)
{
	memset(ts, 0, sizeof(*ts));

	ts->input.component_type = cgltf_component_type_r_32f;
	ts->input.type = cgltf_type_scalar;
	ts->input.count = keyframe_count;
	ts->input.stride = sizeof(cgltf_float);
	ts->input.buffer_view = &ts->input_view;
	ts->input_view.data = (void*)times;

	ts->output.component_type = cgltf_component_type_r_32f;
	ts->output.type = value_type;
	ts->output.count = value_count;
	ts->output.stride = cgltf_num_components(value_type) * sizeof(cgltf_float);
	ts->output.buffer_view = &ts->output_view;
	ts->output_view.data = (void*)values;

	ts->sampler.input = &ts->input;
	ts->sampler.output = &ts->output;
	ts->sampler.interpolation = interpolation;
}

static cgltf_animation_channel make_channel(cgltf_animation_sampler* sampler, cgltf_node* node, cgltf_animation_path_type path)
{
	cgltf_animation_channel channel = {};
	channel.sampler = sampler;
	channel.target_node = node;
	channel.target_path = path;
	return channel;
}

/* quaternion rotation angle in radians, computed in double precision */
static double quat_angle(const cgltf_float* q)
{
	double v = sqrt((double)q[0] * q[0] + (double)q[1] * q[1] + (double)q[2] * q[2]);
	return 2.0 * atan2(v, (double)q[3]);
}

/* rotation angle error between two quaternions: 2*sin(dtheta/2) ~ dtheta, sign-independent */
static double quat_angle_error(const cgltf_float* a, const double* b)
{
	double d_minus = 0, d_plus = 0;
	for (int i = 0; i < 4; ++i)
	{
		double dm = (double)a[i] - b[i];
		double dp = (double)a[i] + b[i];
		d_minus += dm * dm;
		d_plus += dp * dp;
	}
	return sqrt(d_minus < d_plus ? d_minus : d_plus);
}

static void test_linear_translation()
{
	const cgltf_float times[] = {0, 1, 2};
	const cgltf_float values[] = {0, 0, 0, 10, 0, 0, 10, 20, 20};

	TestSampler ts;
	init_sampler(&ts, times, 3, values, cgltf_type_vec3, 3, cgltf_interpolation_type_linear);

	cgltf_node node = {};
	cgltf_animation_channel channel = make_channel(&ts.sampler, &node, cgltf_animation_path_type_translation);

	cgltf_float out[3];
	const struct { double t, x, y, z; } cases[] = {
		{0.0, 0, 0, 0},
		{0.5, 5, 0, 0},
		{1.0, 10, 0, 0},
		{1.25, 10, 5, 5},
		{1.5, 10, 10, 10},
		{1.75, 10, 15, 15},
		{2.0, 10, 20, 20},
	};
	for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
	{
		if (!cgltf_animation_channel_sample(&channel, (cgltf_float)cases[i].t, out))
		{
			fprintf(stderr, "FAIL: linear translation sampling returned false\n");
			failures++;
			continue;
		}
		const double expected[] = {cases[i].x, cases[i].y, cases[i].z};
		check_vec("linear translation", out, expected, 3, 1e-6);
	}
}

static void test_step_scale()
{
	const cgltf_float times[] = {0, 1};
	const cgltf_float values[] = {1, 1, 1, 2, 2, 2};

	TestSampler ts;
	init_sampler(&ts, times, 2, values, cgltf_type_vec3, 2, cgltf_interpolation_type_step);

	cgltf_animation_channel channel = make_channel(&ts.sampler, NULL, cgltf_animation_path_type_scale);

	cgltf_float out[3];
	const double one[] = {1, 1, 1};
	const double two[] = {2, 2, 2};

	cgltf_animation_channel_sample(&channel, 0.0f, out);
	check_vec("step scale t=0", out, one, 3, 1e-6);
	cgltf_animation_channel_sample(&channel, 0.5f, out);
	check_vec("step scale t=0.5", out, one, 3, 1e-6);
	cgltf_animation_channel_sample(&channel, 0.999999f, out);
	check_vec("step scale t=0.999999", out, one, 3, 1e-6);
	cgltf_animation_channel_sample(&channel, 1.0f, out);
	check_vec("step scale t=1", out, two, 3, 1e-6);
}

static void test_cubic_spline_translation()
{
	/* p0=(0,0,0) out-tangent=(2,0,0); p1=(1,2,3) in-tangent=(0,0,0); segment duration 1s.
	   Hermite basis at t: h00=2t^3-3t^2+1, h10=t^3-2t^2+t, h01=3t^2-2t^3, h11=t^3-t^2.
	   t=0.5:  x = 0.125*2 + 0.5*1 = 0.75,   y = 0.5*2 = 1.0,    z = 0.5*3 = 1.5
	   t=0.25: x = 0.140625*2 + 0.15625*1 = 0.4375, y = 0.15625*2 = 0.3125, z = 0.15625*3 = 0.46875 */
	const cgltf_float times[] = {0, 1};
	const cgltf_float values[] = {
		0, 0, 0, /* in-tangent k0 */
		0, 0, 0, /* value k0 */
		2, 0, 0, /* out-tangent k0 */
		0, 0, 0, /* in-tangent k1 */
		1, 2, 3, /* value k1 */
		0, 0, 0, /* out-tangent k1 */
	};

	TestSampler ts;
	init_sampler(&ts, times, 2, values, cgltf_type_vec3, 6, cgltf_interpolation_type_cubic_spline);

	cgltf_animation_channel channel = make_channel(&ts.sampler, NULL, cgltf_animation_path_type_translation);

	cgltf_float out[3];
	const struct { double t, x, y, z; } cases[] = {
		{0.0, 0, 0, 0},
		{0.25, 0.4375, 0.3125, 0.46875},
		{0.5, 0.75, 1.0, 1.5},
		{1.0, 1, 2, 3},
	};
	for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
	{
		cgltf_animation_channel_sample(&channel, (cgltf_float)cases[i].t, out);
		const double expected[] = {cases[i].x, cases[i].y, cases[i].z};
		check_vec("cubic spline translation", out, expected, 3, 1e-6);
	}
}

static void test_linear_rotation_slerp()
{
	const double h = 0.7071067811865476; /* sqrt(2)/2 */
	const cgltf_float times[] = {0, 1};
	const cgltf_float values[] = {0, 0, 0, 1, 0, 0, (cgltf_float)h, (cgltf_float)h}; /* identity -> 90 deg about z */

	TestSampler ts;
	init_sampler(&ts, times, 2, values, cgltf_type_vec4, 2, cgltf_interpolation_type_linear);

	cgltf_animation_channel channel = make_channel(&ts.sampler, NULL, cgltf_animation_path_type_rotation);

	const double s22 = 0.3826834323650898; /* sin(22.5 deg) */
	const double c22 = 0.9238795325112867; /* cos(22.5 deg) */

	cgltf_float out[4];
	cgltf_animation_channel_sample(&channel, 0.5f, out);

	/* expected: 45 degrees about z = (0, 0, sin(22.5), cos(22.5)) */
	const double expected[] = {0, 0, s22, c22};
	check_vec("slerp midpoint", out, expected, 4, 1e-6);

	/* angle of the sampled quaternion must be 45 degrees */
	check_near("slerp midpoint angle", quat_angle(out), 0.7853981633974483, 1e-6);

	/* angle error metric vs expected quaternion */
	check_near("slerp midpoint angle error", quat_angle_error(out, expected), 0.0, 1e-6);

	/* quarter of the way: 22.5 degrees about z */
	cgltf_animation_channel_sample(&channel, 0.25f, out);
	const double expected_q[] = {0, 0, 0.19509032201612825, 0.9807852804032304};
	check_vec("slerp quarter", out, expected_q, 4, 1e-6);
	check_near("slerp quarter angle", quat_angle(out), 0.39269908169872414, 1e-6);

	/* negated second quaternion encodes the same rotation; slerp must take the short path */
	const cgltf_float neg_values[] = {0, 0, 0, 1, 0, 0, (cgltf_float)-h, (cgltf_float)-h};
	TestSampler ts2;
	init_sampler(&ts2, times, 2, neg_values, cgltf_type_vec4, 2, cgltf_interpolation_type_linear);
	cgltf_animation_channel channel2 = make_channel(&ts2.sampler, NULL, cgltf_animation_path_type_rotation);
	cgltf_animation_channel_sample(&channel2, 0.5f, out);
	check_near("slerp negated hemisphere angle", quat_angle(out), 0.7853981633974483, 1e-6);
	check_near("slerp negated hemisphere angle error", quat_angle_error(out, expected), 0.0, 1e-6);
}

static void test_cubic_spline_rotation_normalized()
{
	/* zero tangents: result is h00*q0 + h01*q1, which is not unit length before normalization */
	const double h = 0.7071067811865476;
	const cgltf_float times[] = {0, 1};
	const cgltf_float values[] = {
		0, 0, 0, 0,
		0, 0, 0, 1,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, (cgltf_float)h, (cgltf_float)h,
		0, 0, 0, 0,
	};

	TestSampler ts;
	init_sampler(&ts, times, 2, values, cgltf_type_vec4, 6, cgltf_interpolation_type_cubic_spline);

	cgltf_animation_channel channel = make_channel(&ts.sampler, NULL, cgltf_animation_path_type_rotation);

	cgltf_float out[4];
	cgltf_animation_channel_sample(&channel, 0.5f, out);

	/* hand-computed: 0.5*(0,0,0,1) + 0.5*(0,0,h,h), normalized */
	double raw[4] = {0, 0, 0.5 * h, 0.5 + 0.5 * h};
	double len = sqrt(raw[2] * raw[2] + raw[3] * raw[3]);
	const double expected[] = {0, 0, raw[2] / len, raw[3] / len};
	check_vec("cubic spline rotation", out, expected, 4, 1e-6);

	double norm = 0;
	for (int i = 0; i < 4; ++i)
		norm += (double)out[i] * out[i];
	check_near("cubic spline rotation unit length", sqrt(norm), 1.0, 1e-6);
}

static void test_clamping()
{
	const cgltf_float times[] = {1, 2, 3};
	const cgltf_float values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

	TestSampler ts;
	init_sampler(&ts, times, 3, values, cgltf_type_vec3, 3, cgltf_interpolation_type_linear);

	cgltf_animation_channel channel = make_channel(&ts.sampler, NULL, cgltf_animation_path_type_translation);

	cgltf_float out[3];
	const double first[] = {1, 2, 3};
	const double last[] = {7, 8, 9};

	cgltf_animation_channel_sample(&channel, -100.0f, out);
	check_vec("clamp below range", out, first, 3, 1e-6);
	cgltf_animation_channel_sample(&channel, 0.999999f, out);
	check_vec("clamp just below range", out, first, 3, 1e-5);
	cgltf_animation_channel_sample(&channel, 100.0f, out);
	check_vec("clamp above range", out, last, 3, 1e-6);
	cgltf_animation_channel_sample(&channel, 3.000001f, out);
	check_vec("clamp just above range", out, last, 3, 1e-5);
}

static void test_multi_channel_animation()
{
	const cgltf_float t_times[] = {0, 2};
	const cgltf_float t_values[] = {0, 0, 0, 4, 6, 8};
	const cgltf_float r_times[] = {0, 2};
	const double h = 0.7071067811865476;
	const cgltf_float r_values[] = {0, 0, 0, 1, 0, 0, (cgltf_float)h, (cgltf_float)h};
	const cgltf_float s_times[] = {0, 2};
	const cgltf_float s_values[] = {1, 1, 1, 2, 4, 8};
	const cgltf_float w_times[] = {0, 2};
	const cgltf_float w_values[] = {0, 1, 1, 0}; /* 2 morph weights per keyframe */

	TestSampler tts, rts, sts, wts;
	init_sampler(&tts, t_times, 2, t_values, cgltf_type_vec3, 2, cgltf_interpolation_type_linear);
	init_sampler(&rts, r_times, 2, r_values, cgltf_type_vec4, 2, cgltf_interpolation_type_linear);
	init_sampler(&sts, s_times, 2, s_values, cgltf_type_vec3, 2, cgltf_interpolation_type_step);
	init_sampler(&wts, w_times, 2, w_values, cgltf_type_scalar, 4, cgltf_interpolation_type_linear);

	cgltf_node node = {};
	cgltf_float weights[2] = {0, 0};
	node.weights = weights;
	node.weights_count = 2;

	cgltf_animation_channel channels[4];
	channels[0] = make_channel(&tts.sampler, &node, cgltf_animation_path_type_translation);
	channels[1] = make_channel(&rts.sampler, &node, cgltf_animation_path_type_rotation);
	channels[2] = make_channel(&sts.sampler, &node, cgltf_animation_path_type_scale);
	channels[3] = make_channel(&wts.sampler, &node, cgltf_animation_path_type_weights);

	cgltf_animation animation = {};
	animation.channels = channels;
	animation.channels_count = 4;

	cgltf_size applied = cgltf_animation_sample(&animation, 1.0f);
	check_near("multi-channel applied count", (double)applied, 4, 0);

	const double et[] = {2, 3, 4};
	check_vec("multi-channel translation", node.translation, et, 3, 1e-6);
	check_near("multi-channel has_translation", node.has_translation, 1, 0);

	const double er[] = {0, 0, 0.3826834323650898, 0.9238795325112867};
	check_vec("multi-channel rotation", node.rotation, er, 4, 1e-6);
	check_near("multi-channel rotation angle", quat_angle(node.rotation), 0.7853981633974483, 1e-6);

	const double es[] = {1, 1, 1}; /* step holds the first keyframe */
	check_vec("multi-channel scale", node.scale, es, 3, 1e-6);

	const double ew[] = {0.5, 0.5};
	check_vec("multi-channel weights", node.weights, ew, 2, 1e-6);

	/* out-of-range time clamps all channels */
	cgltf_animation_sample(&animation, 100.0f);
	const double et2[] = {4, 6, 8};
	check_vec("multi-channel clamp translation", node.translation, et2, 3, 1e-6);
	const double es2[] = {2, 4, 8};
	check_vec("multi-channel clamp scale", node.scale, es2, 3, 1e-6);
}

static void test_performance()
{
	const cgltf_size joint_count = 1000;
	const cgltf_size channels_per_joint = 3; /* translation + rotation + scale */
	const cgltf_size keyframe_count = 10;

	std::vector<cgltf_float> times(keyframe_count);
	for (cgltf_size i = 0; i < keyframe_count; ++i)
		times[i] = (cgltf_float)i;

	/* per-joint keyframe data, deterministic pseudo-random values */
	std::vector<cgltf_float> values(joint_count * channels_per_joint * keyframe_count * 4);
	unsigned int rng = 12345;
	for (size_t i = 0; i < values.size(); ++i)
	{
		rng = rng * 1664525u + 1013904223u;
		values[i] = (cgltf_float)((rng >> 8) & 0xffff) / 65536.0f;
	}

	std::vector<cgltf_node> nodes(joint_count);
	std::vector<cgltf_animation_channel> channels(joint_count * channels_per_joint);
	std::vector<cgltf_animation_sampler> samplers(joint_count * channels_per_joint);
	std::vector<cgltf_accessor> inputs(joint_count * channels_per_joint);
	std::vector<cgltf_accessor> outputs(joint_count * channels_per_joint);
	std::vector<cgltf_buffer_view> input_views(joint_count * channels_per_joint);
	std::vector<cgltf_buffer_view> output_views(joint_count * channels_per_joint);

	for (cgltf_size j = 0; j < joint_count; ++j)
	{
		for (cgltf_size c = 0; c < channels_per_joint; ++c)
		{
			cgltf_size idx = j * channels_per_joint + c;
			cgltf_bool is_rotation = (c == 1);
			cgltf_size comps = is_rotation ? 4 : 3;
			cgltf_float* v = &values[idx * keyframe_count * 4];

			cgltf_accessor* input = &inputs[idx];
			input->component_type = cgltf_component_type_r_32f;
			input->type = cgltf_type_scalar;
			input->count = keyframe_count;
			input->stride = sizeof(cgltf_float);
			input->buffer_view = &input_views[idx];
			input_views[idx].data = &times[0];

			cgltf_accessor* output = &outputs[idx];
			output->component_type = cgltf_component_type_r_32f;
			output->type = is_rotation ? cgltf_type_vec4 : cgltf_type_vec3;
			output->count = keyframe_count;
			output->stride = comps * sizeof(cgltf_float);
			output->buffer_view = &output_views[idx];
			output_views[idx].data = v;

			cgltf_animation_sampler* sampler = &samplers[idx];
			sampler->input = input;
			sampler->output = output;
			sampler->interpolation = cgltf_interpolation_type_linear;

			cgltf_animation_channel* channel = &channels[idx];
			channel->sampler = sampler;
			channel->target_node = &nodes[j];
			channel->target_path = c == 0 ? cgltf_animation_path_type_translation
				: (c == 1 ? cgltf_animation_path_type_rotation : cgltf_animation_path_type_scale);
		}
	}

	cgltf_animation animation = {};
	animation.channels = &channels[0];
	animation.channels_count = channels.size();

	/* warmup */
	cgltf_size applied = cgltf_animation_sample(&animation, 3.7f);
	check_near("performance applied count", (double)applied, (double)channels.size(), 0);

	const int iterations = 100;
	double best_ms = 1e30, total_ms = 0;
	for (int it = 0; it < iterations; ++it)
	{
		std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		cgltf_animation_sample(&animation, 3.7f);
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		double ms = std::chrono::duration<double, std::milli>(end - start).count();
		total_ms += ms;
		if (ms < best_ms)
			best_ms = ms;
	}

	printf("performance: 1000 joints x 3 channels, avg %.4f ms, best %.4f ms over %d iterations\n",
		total_ms / iterations, best_ms, iterations);

	if (best_ms >= 5.0)
	{
		fprintf(stderr, "FAIL: sampling 1000 joints took %.4f ms (>= 5 ms)\n", best_ms);
		failures++;
	}
}

int main(int, char**)
{
	test_linear_translation();
	test_step_scale();
	test_cubic_spline_translation();
	test_linear_rotation_slerp();
	test_cubic_spline_rotation_normalized();
	test_clamping();
	test_multi_channel_animation();
	test_performance();

	if (failures)
	{
		fprintf(stderr, "%d check(s) failed\n", failures);
		return 1;
	}
	printf("all animation sampling tests passed\n");
	return 0;
}
