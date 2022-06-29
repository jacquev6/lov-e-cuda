#!/usr/bin/env python3

import multiprocessing
import re
import subprocess


def main():
    regenerate(
        "lov-e.hpp",
        re.compile(r"// (BEGIN|END) GENERATED SECTION: (.*)"),
        {class_.key: class_ for class_ in [ArraysAndArrayViewSection]},
    )
    subprocess.run(["./make.sh", f"-j{multiprocessing.cpu_count()}"], check=True)
    subprocess.run(["./make.sh", "-j1", "examples"], check=True)
    regenerate(
        "README.md",
        re.compile(r"<!-- (BEGIN|END) GENERATED SECTION: (.*) -->"),
        {class_.key: class_ for class_ in [ExamplesPerformanceTableSection]},
    )


def regenerate(file_name, generated_re, section_classes):
    with open(file_name) as f:
        original_lines = [line.rstrip() for line in f.readlines()]

    def gen():
        sections = [HandWrittenSection()]
        for line in original_lines:
            m = generated_re.fullmatch(line)
            if m:
                if m.group(1) == "BEGIN":
                    sections.append(section_classes[m.group(2)]())
                    sections[-1].entry = m.group(0)
                elif m.group(1) == "END":
                    sections[-1].exit = m.group(0)
                    sections.append(HandWrittenSection())
                else:
                    assert False
            else:
                sections[-1].record_original_line(line)

        for section in sections:
            if section.key is not None:
                yield section.entry
                yield ""
            for line in section.generate():
                yield line
            if section.key is not None:
                yield ""
                yield section.exit

    new_lines = list(gen())
    if new_lines != original_lines:
        with open(file_name, "w") as f:
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
            with open(f"build/release/examples/{base_name}.log") as f:
                (line,) = f.readlines()
                m = re.fullmatch(r"Mandelbrot set computed in (.*) s, at (.*) Mpix/s", line.rstrip())
                return f"{int(float(m.group(1)) * 1000)} ms *i.e.* {int(float(m.group(2)))} Mpix/s"

        for (description, name, parse) in [
            ("Mandelbrot<br>(static parallelism)", "mandelbrot", parse_mandelbrot),
            ("Mandelbrot<br>(dynamic parallelism)", "mandelbrot-dyn", parse_mandelbrot)
        ]:
            yield f"| {description} | {parse(name)} | {parse(name + '-lov-e')} |"


class ArraysAndArrayViewSection:
    key = "arrays-and-array-views"

    def record_original_line(self, _):
        pass

    def generate(self):
        first = True
        for n in range(1, 6):
            if first:
                first = False
            else:
                yield ""
            yield from self.generate_one(n)

    def generate_one(self, n):
        ds = list(range(n - 1, -1, -1))
        lower_ds = ds[1:]

        def sep(f, sep=', ', ds=ds):
            return sep.join(f.format(d=d) for d in ds)

        if n == 1:
            pass
        else:
            yield f"template<typename Where, typename T> class Array{n}D;"
            yield ""
            yield "template<typename Where, typename T>"
            yield f"class ArrayView{n}D {{"
            yield " public:"
            yield "  // Constructor"
            yield "  HOST_DEVICE_DECORATORS"
            yield f"  ArrayView{n}D({sep('std::size_t s{d}')}, T* data) :"
            yield f"    {sep('_s{d}(s{d})')}, _data(data) {{}}"
            yield ""
            yield '  // No need for custom copy and move constructors and operators (cf. "Rule Of Zero" above)'
            yield ""
            yield "  // Generalized copy constructor and operator"
            yield "  template<typename U>"
            yield "  HOST_DEVICE_DECORATORS"
            yield f"  ArrayView{n}D(const ArrayView{n}D<Where, U>& o) :"
            yield f"    {sep('_s{d}(o.s{d}())')}, _data(o.data_for_legacy_use()) {{}}"
            yield ""
            yield "  template<typename U>"
            yield "  HOST_DEVICE_DECORATORS"
            yield f"  ArrayView{n}D& operator=(const ArrayView{n}D<Where, U>& o) {{"
            for d in ds:
                yield f"    _s{d} = o.s{d}();"
            yield "    _data = o.data_for_legacy_use();"
            yield "    return *this;"
            yield "  }"
            yield ""
            yield "  // Accessors"
            for d in ds:
                yield "  HOST_DEVICE_DECORATORS"
                yield f"  std::size_t s{d}() const {{ return _s{d}; }}"
            yield ""
            yield "  HOST_DEVICE_DECORATORS"
            yield f"  ArrayView{n-1}D<Where, T> operator[](unsigned i{n-1}) const {{"
            yield f"    assert(i{n-1} < _s{n-1});"
            yield f"    return ArrayView{n-1}D<Where, T>({sep('_s{d}', ', ', lower_ds)}, _data + i{n-1} * {sep('_s{d}', ' * ', lower_ds)});"
            yield "  }"
            yield ""
            yield "  HOST_DEVICE_DECORATORS"
            yield "  const T* data_for_legacy_use() const { return _data; }"
            yield "  HOST_DEVICE_DECORATORS"
            yield "  T* data_for_legacy_use() { return _data; }"
            yield ""
            yield "  // Clonable"
            yield "  template<typename WhereTo>"
            yield f"  Array{n}D<WhereTo, T> clone_to() const;"
            yield ""
            yield " private:"
            for d in ds:
                yield f"  std::size_t _s{d};"
            yield "  T* _data;"
            yield ""
            yield f"  friend class Array{n}D<Where, T>;"
            yield "};"

            yield ""

        yield "template<typename WhereFrom, typename WhereTo, typename T>"
        yield f"void copy(ArrayView{n}D<WhereFrom, T> src, ArrayView{n}D<WhereTo, T> dst) {{"
        for d in ds:
            yield f"  assert(dst.s{d}() == src.s{d}());"
        yield ""
        yield "  From<WhereFrom>::template To<WhereTo>::template copy("
        yield f"    {sep('src.s{d}()', ' * ')}, src.data_for_legacy_use(), dst.data_for_legacy_use());"
        yield "}"

        yield ""

        yield "template<typename Where, typename T>"
        yield f"class Array{n}D : public ArrayView{n}D<Where, T> {{"
        yield " public:"
        yield "  // RAII"
        yield f"  Array{n}D({sep('std::size_t s{d}')}, Uninitialized) :"
        yield f"    ArrayView{n}D<Where, T>({sep('s{d}')}, Where::template alloc<T>({sep('s{d}', ' * ')}))"
        yield "  {}"
        yield f"  Array{n}D({sep('std::size_t s{d}')}, Zeroed) :"
        yield f"    ArrayView{n}D<Where, T>({sep('s{d}')}, Where::template alloc_zeroed<T>({sep('s{d}', ' * ')}))"
        yield "  {}"
        yield f"  ~Array{n}D() {{"
        yield "    free();"
        yield "  }"
        yield ""
        yield "  // Not copyable"
        yield f"  Array{n}D(const Array{n}D&) = delete;"
        yield f"  Array{n}D& operator=(const Array{n}D&) = delete;"
        yield ""
        yield "  // But movable"
        yield f"  Array{n}D(Array{n}D&& o) : ArrayView{n}D<Where, T>(o) {{"
        for d in ds:
            yield f"    o._s{d} = 0;"
        yield "    o._data = nullptr;"
        yield "  }"
        yield f"  Array{n}D& operator=(Array{n}D&& o) {{"
        yield "    free();"
        yield f"    static_cast<ArrayView{n}D<Where, T>&>(*this) = o;"
        for d in ds:
            yield f"    o._s{d} = 0;"
        yield "    o._data = nullptr;"
        yield "    return *this;"
        yield "  }"
        yield ""
        yield " private:"
        yield "  void free() {"
        yield "    Where::free(this->_data);"
        yield "  }"
        yield "};"

        yield ""

        if n == 1:
            yield "template<typename T>"
            yield "template<typename WhereTo>"
            yield f"Array{n}D<WhereTo, T> ArrayView{n}D<Host, T>::clone_to() const {{"
            yield f"  Array{n}D<WhereTo, T> dst(this->s0(), uninitialized);"
            yield "  copy(*this, dst);  // NOLINT(build/include_what_you_use)"
            yield "  return dst;"
            yield "}"
            yield ""
            yield "template<typename T>"
            yield "template<typename WhereTo>"
            yield f"Array{n}D<WhereTo, T> ArrayView{n}D<Device, T>::clone_to() const {{"
            yield f"  Array{n}D<WhereTo, T> dst(this->s0(), uninitialized);"
            yield "  copy(*this, dst);  // NOLINT(build/include_what_you_use)"
            yield "  return dst;"
            yield "}"
        else:
            yield "template<typename WhereFrom, typename T>"
            yield "template<typename WhereTo>"
            yield f"Array{n}D<WhereTo, T> ArrayView{n}D<WhereFrom, T>::clone_to() const {{"
            yield f"  Array{n}D<WhereTo, T> dst({sep('this->s{d}()')}, uninitialized);"
            yield "  copy(*this, dst);  // NOLINT(build/include_what_you_use)"
            yield "  return dst;"
            yield "}"


if __name__ == "__main__":
    main()
