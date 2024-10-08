import cv2
import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import output_file, show
from bokeh.layouts import layout, gridplot
from bokeh.models import Div
from bokeh.plotting import figure, ColumnDataSource


def put_text(
    image, org, text, color=(0, 0, 255), fontScale=0.7, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX
):
    if not isinstance(org, tuple):
        (label_width, label_height), baseline = cv2.getTextSize(text, font, fontScale, thickness)
        org_w = 0
        org_h = 0

        h, w, *_ = image.shape

        place_h, place_w = org.split("_")

        if place_h == "top":
            org_h = label_height
        elif place_h == "bottom":
            org_h = h
        elif place_h == "center":
            org_h = h // 2 + label_height // 2

        if place_w == "left":
            org_w = 0
        elif place_w == "right":
            org_w = w - label_width
        elif place_w == "center":
            org_w = w // 2 - label_width // 2

        org = (org_w, org_h)

    image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    return image


def draw_matches(img1, kp1, img2, kp2, matches):
    matches = sorted(matches, key=lambda x: x.distance)
    vis_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis_img


def show_images(images, window_name="Image", image_title=None):
    if len(images.shape) == 2:
        images = [images]

    for i, image in enumerate(images):
        image_c = image.copy()

        if image_c.dtype != np.uint8:
            if image_c.max() < 1.0:
                image_c = image_c * 255
            image_c = image_c.astype(np.uint8)

        if len(image.shape) == 2:
            image_c = cv2.cvtColor(image_c, cv2.COLOR_GRAY2BGR)

        if image_title is None:
            image_title_show = f"{i}"
        else:
            image_title_show = image_title

        image_c = put_text(image_c, "top_center", image_title_show)
        cv2.imshow(window_name, image_c)
        cv2.waitKey(0)


def visualize_paths(
    gt_path, pred_path, html_title="", title="Visual Odometry", file_out="plot.html"
):
    output_file(file_out, title=html_title)
    gt_path = np.array(gt_path)
    pred_path = np.array(pred_path)

    tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

    gt_x, gt_y = gt_path.T
    pred_x, pred_y = pred_path.T
    xs = list(np.array([gt_x, pred_x]).T)
    ys = list(np.array([gt_y, pred_y]).T)

    diff = np.linalg.norm(gt_path - pred_path, axis=1)
    source = ColumnDataSource(
        data=dict(
            gtx=gt_path[:, 0],
            gty=gt_path[:, 1],
            px=pred_path[:, 0],
            py=pred_path[:, 1],
            diffx=np.arange(len(diff)),
            diffy=diff,
            disx=xs,
            disy=ys,
        )
    )

    fig1 = figure(
        title="Paths",
        tools=tools,
        match_aspect=True,
        width_policy="max",
        toolbar_location="above",
        x_axis_label="x",
        y_axis_label="y",
    )
    fig1.circle(
        "gtx", "gty", source=source, color="blue", hover_fill_color="firebrick", legend_label="GT"
    )
    fig1.line("gtx", "gty", source=source, color="blue", legend_label="GT")

    fig1.circle(
        "px", "py", source=source, color="green", hover_fill_color="firebrick", legend_label="Pred"
    )
    fig1.line("px", "py", source=source, color="green", legend_label="Pred")

    fig1.multi_line(
        "disx", "disy", source=source, legend_label="Error", color="red", line_dash="dashed"
    )
    fig1.legend.click_policy = "hide"

    fig2 = figure(
        title="Error",
        tools=tools,
        width_policy="max",
        toolbar_location="above",
        x_axis_label="frame",
        y_axis_label="error",
    )
    fig2.circle("diffx", "diffy", source=source, hover_fill_color="firebrick", legend_label="Error")
    fig2.line("diffx", "diffy", source=source, legend_label="Error")

    show(
        layout(
            [
                Div(text=f"<h1>{title}</h1>"),
                Div(text="<h2>Paths</h1>"),
                [fig1, fig2],
            ],
            sizing_mode="scale_width",
        )
    )


def make_residual_plot(x, residual_init, residual_minimized):
    fig1 = figure(
        title="Initial residuals",
        x_range=[0, len(residual_init)],
        x_axis_label="residual",
        y_axis_label="",
    )
    fig1.line(x, residual_init)

    change = np.abs(residual_minimized) - np.abs(residual_init)
    plot_data = ColumnDataSource(data={"x": x, "residual": residual_minimized, "change": change})
    tooltips = [
        ("change", "@change"),
    ]
    fig2 = figure(
        title="Optimized residuals",
        x_axis_label=fig1.xaxis.axis_label,
        y_axis_label=fig1.yaxis.axis_label,
        x_range=fig1.x_range,
        y_range=fig1.y_range,
        tooltips=tooltips,
    )
    fig2.line("x", "residual", source=plot_data)

    fig3 = figure(
        title="Change",
        x_axis_label=fig1.xaxis.axis_label,
        y_axis_label=fig1.yaxis.axis_label,
        x_range=fig1.x_range,
        tooltips=tooltips,
    )
    fig3.line("x", "change", source=plot_data)
    return fig1, fig2, fig3


def plot_residual_results(
    qs_small, small_residual_init, small_residual_minimized, qs, residual_init, residual_minimized
):
    output_file("plot.html", title="Bundle Adjustment")
    x = np.arange(2 * qs_small.shape[0])
    fig1, fig2, fig3 = make_residual_plot(x, small_residual_init, small_residual_minimized)

    x = np.arange(2 * qs.shape[0])
    fig4, fig5, fig6 = make_residual_plot(x, residual_init, residual_minimized)

    show(
        layout(
            [
                Div(text="<h1>Bundle Adjustment exercises</h1>"),
                Div(text="<h2>Bundle adjustment with reduced parameters</h1>"),
                gridplot([[fig1, fig2, fig3]], toolbar_location="above"),
                Div(text="<h2>Bundle adjustment with all parameters (with sparsity)</h1>"),
                gridplot([[fig4, fig5, fig6]], toolbar_location="above"),
            ]
        )
    )


def plot_sparsity(sparse_mat):
    fig, ax = plt.subplots(figsize=[20, 10])
    plt.title("Sparsity matrix")

    ax.spy(sparse_mat, aspect="auto", markersize=0.02)
    plt.xlabel("Parameters")
    plt.ylabel("Resudals")

    plt.show()


def play_trip(
    l_frames,
    r_frames=None,
    lat_lon=None,
    timestamps=None,
    color_mode=False,
    waite_time=100,
    win_name="Trip",
):
    l_r_mode = r_frames is not None

    if not l_r_mode:
        r_frames = [None] * len(l_frames)

    frame_count = 0
    for i, frame_step in enumerate(zip(l_frames, r_frames)):
        img_l, img_r = frame_step

        if not color_mode:
            img_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
            if img_r is not None:
                img_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)

        if img_r is not None:
            img_l = put_text(img_l, "top_center", "Left")
            img_r = put_text(img_r, "top_center", "Right")
            show_image = np.vstack([img_l, img_r])
        else:
            show_image = img_l
        show_image = put_text(show_image, "top_left", "Press ESC to stop")
        show_image = put_text(show_image, "top_right", f"Frame: {frame_count}/{len(l_frames)}")

        if timestamps is not None:
            time = timestamps[i]
            show_image = put_text(show_image, "bottom_right", f"{time}")

        if lat_lon is not None:
            lat, lon = lat_lon[i]
            show_image = put_text(show_image, "bottom_left", f"{lat}, {lon}")

        cv2.imshow(win_name, show_image)

        key = cv2.waitKey(waite_time)
        if key == 27:  # ESC
            break
        frame_count += 1
    cv2.destroyWindow(win_name)


def draw_matches_frame(img1, kp1, img2, kp2, matches):
    """
    Need to be call for each frame
    """
    matches = sorted(matches, key=lambda x: x.distance)
    vis_img = draw_matches(img1, kp1, img2, kp2, matches)
    cv2.imshow("Matches", vis_img)
    cv2.waitKey(100)
