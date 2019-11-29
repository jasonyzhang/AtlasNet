import sys
sys.path.append('/home/thibault/ssd/netvision/')
import HtmlGenerator
import os
from copy import deepcopy
import auxiliary.init_html_report as init_html_report


def main(trainer, outHtml=None):
    if outHtml is None:
        outHtml = os.path.join(trainer.opt.dir_name, f"{trainer.epoch}") + '.html'
    else:
        outHtml = os.path.join(trainer.opt.dir_name, outHtml)

    webpage = HtmlGenerator.HtmlGenerator(path=outHtml, title=trainer.opt.dir_name, local_copy=True)
    loss_val = trainer.log.meters["loss_val"].avg
    webpage.add_title(f"Reconstruction (Val):{loss_val} Metro:{trainer.metro_results} Fscore:{trainer.html_report_data.fscore_curve[-1]}")

    table = webpage.add_table()

    table2 = deepcopy(table)
    table2.add_titleless_columns(1)
    table.add_titleless_columns(2)
    curve_recons = webpage.chart(trainer.html_report_data.data_curve, title="Reconstruction quality - chamfer log")
    table2.add_row([curve_recons])
    curve_recons = webpage.chart(trainer.html_report_data.fscore_curve, title="Reconstruction quality - fscore")
    table2.add_row([curve_recons])

    table.add_row([webpage.dict(trainer.opt), table2], "")
    for i in range(3):
        output_mesh = trainer.html_report_data.output_meshes[i]
        table.add_row([webpage.image(output_mesh["image_path"]),
                       webpage.mesh(output_mesh["output_path"], normalize=True)], "")



    webpage.return_html()



    # Add main results in master webpage
    if not os.path.exists("master.pkl"):
        init_html_report.main()
    webpage_after = HtmlGenerator.HtmlGenerator(path="master.html", reload_path="master.pkl")
    webpage_after.tables[trainer.opt.dataset].add_row(
        [f"{trainer.opt.nb_primitives}", f"{trainer.opt.template_type}", f"{loss_val}", trainer.opt.dir_name])
    webpage_after.return_html(save_editable_version=True)