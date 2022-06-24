#!/usr/bin/env python3

import glob
import multiprocessing
import os
import re
import subprocess


def main():
    subprocess.run(["./make.sh", f"-j{multiprocessing.cpu_count()}"], check=True)
    subprocess.run(["./make.sh", "-j1", "examples"], check=True)
    regenerate_readme()


def regenerate_readme():
    with open("README.md") as f:
        original_lines = [line.rstrip() for line in f.readlines()]

    def gen():
        generated_re = re.compile(r"<!-- (BEGIN|END) GENERATED SECTION: (.*) -->")

        sections = [HandWrittenSection()]
        for line in original_lines:
            m = generated_re.fullmatch(line)
            if m:
                if m.group(1) == "BEGIN":
                    sections.append(section_classes[m.group(2)]())
                elif m.group(1) == "END":
                    sections.append(HandWrittenSection())
                else:
                    assert False
            else:
                sections[-1].record_original_line(line)

        for section in sections:
            if section.key is not None:
                yield f"<!-- BEGIN GENERATED SECTION: {section.key} -->"
            for line in section.generate():
                yield line
            if section.key is not None:
                yield f"<!-- END GENERATED SECTION: {section.key} -->"

    new_lines = list(gen())
    if new_lines != original_lines:
        with open("README.md", "w") as f:
            for line in new_lines:
                f.write(f"{line}\n")


class HandWrittenSection:
    key = None

    def __init__(self):
        self.__original_lines = []

    def record_original_line(self, line):
        self.__original_lines.append(line)

    def generate(self):
        return self.__original_lines


class ExamplesPerformanceTableSection:
    key = "examples-performance-table"

    def record_original_line(self, _):
        pass

    def generate(self):
        yield "| Example | Without *Lov-e-cuda* | With *Lov-e-cuda* |"
        yield "| --- | --- | --- |"

        # @todo Average a few executions (10?) to get more significant numbers

        def parse_mandelbrot(base_name):
            with open(f"build/release/examples/{base_name}.ok.log") as f:
                (line,) = f.readlines()
                m = re.fullmatch(r"Mandelbrot set computed in (.*) s, at (.*) Mpix/s", line.rstrip())
                return f"{int(float(m.group(1)) * 1000)} ms *i.e.* {int(float(m.group(2)))} Mpix/s"

        for (description, name, parse) in [
            ("Mandelbrot<br>(static parallelism)", "mandelbrot", parse_mandelbrot),
            ("Mandelbrot<br>(dynamic parallelism)", "mandelbrot-dyn", parse_mandelbrot)
        ]:
            yield f"| {description} | {parse(name)} | {parse(name + '-lov-e')} |"



section_classes = {class_.key: class_ for class_ in [ExamplesPerformanceTableSection]}


if __name__ == "__main__":
    main()
