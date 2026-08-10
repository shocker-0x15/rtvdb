# rtvdb

![Demo](media/demo.webp)

[`rtvdb`](https://github.com/shocker-0x15/rtvdb) は、クライアントアプリからビューワーへ 3D デバッグ描画を送るためのシングルヘッダーライブラリです。\
[`rtvdb`](https://github.com/shocker-0x15/rtvdb) is a single-header library for sending 3D debug drawing from client applications to a viewer.

デフォルトではクライアントとビューワーは同一マシン上で動作させますが、別マシンで動作させることもできます。\
By default, the client and viewer run on the same machine, but they can also run on separate machines.

このプロジェクトは [vdb: the printf of visual debugging (zdevito)](https://github.com/zdevito/vdb) から着想を得て作られています。\
This project was inspired by [vdb: the printf of visual debugging (zdevito)](https://github.com/zdevito/vdb).

## Examples

クライアント組み込み例 / Client integration example
```cpp
#define RTVDB_IMPLEMENTATION
#include "rtvdb/rtvdb.h"

int main() {
    rtvdb::set_color(0.0f, 0.5f, 1.0f);
    rtvdb::triangle(
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        0.0f, 1.0f, 0.0f);

    rtvdb::set_color(1.0f, 0.5f, 0.0f, 0.5f);
    rtvdb::triangle(
        -1.5f, 0.0f, 0.5f,
        1.5f, -0.5f, 0.5f,
        1.0f, 1.0f, 0.5f);

    rtvdb::set_color(0.5f, 1.0f, 0.0f);
    rtvdb::set_line_radius(0.025f);
    rtvdb::line(-1.0f, -0.5f, 0.25f, 1.0f, 0.5f, 0.25f);

    rtvdb::set_color(1.0f, 0.0f, 0.0f);
    rtvdb::set_point_radius(0.05f);
    rtvdb::point(-0.5f, 0.5f, -0.5f);
    rtvdb::point(0.5f, -0.5f, 0.5f);

    return 0;
}
```

ビューワー側の表示 / Viewer output
<p>
    <img src="media/demo_view_a.png" alt="DemoView A" width="360">
    <img src="media/demo_view_b.png" alt="DemoView B" width="360">
</p>

## API

```cpp
bool triangle(
    float ax, float ay, float az,
    float bx, float by, float bz,
    float cx, float cy, float cz,
    std::uint32_t user_data = 0);

void set_point_radius(float value);
bool point(
    float x, float y, float z,
    std::uint32_t user_data = 0);

void set_line_radius(float value);
bool line(
    float ax, float ay, float az,
    float bx, float by, float bz,
    std::uint32_t user_data = 0);

void set_color(float r, float g, float b, float a = 0.0f);
```

- 主 API は raw `float` 引数ですが、`triangle()` / `point()` / `line()` は `.x/.y/.z` を持つ任意型も受け取れます。\
The main API uses raw `float` arguments, but `triangle()`, `point()`, and `line()` also accept any type with `.x`, `.y`, and `.z` members.

- `set_color()` は `.r/.g/.b/.a` を持つ任意型、および `.r/.g/.b` を持つ任意型 + alpha 指定の overload を受け付けます。\
`set_color()` accepts any type with `.r`, `.g`, `.b`, and `.a` members, as well as any type with `.r`, `.g`, and `.b` members when alpha is supplied separately.

### Other Calls

```cpp
bool connect(const config* cfg = nullptr, const char* app_name = kImplicitAppName); // Use this to explicitly specify the destination or app_name
void disconnect(); // Close the connection and reset client-side state
bool is_connected(); // Check the current connection state

bool clear(); // Clear the current scene contents
bool flush(); // Send pending primitive batches immediately

// An explicit frame is displayed atomically. end_frame() requires begin_frame().
// The viewer does not publish an explicit frame until it receives end_frame().
bool begin_frame();
bool end_frame();

// Set a perspective camera
bool set_perspective_camera(
    float origin_x, float origin_y, float origin_z,
    float target_x, float target_y, float target_z,
    float up_x, float up_y, float up_z,
    float vertical_fov_degrees);

// Set a fisheye camera
bool set_fisheye_camera(
    float origin_x, float origin_y, float origin_z,
    float target_x, float target_y, float target_z,
    float up_x, float up_y, float up_z,
    float theta_degrees,
    float phi_degrees);

// Set an orthographic camera
bool set_orthographic_camera(
    float origin_x, float origin_y, float origin_z,
    float target_x, float target_y, float target_z,
    float up_x, float up_y, float up_z,
    float height);

// Set a layer for the section enclosed by push/pop
bool push_layer(const char* name);
bool pop_layer();

// Select the viewer reference grid and coordinate axes
enum class reference_grid : std::uint32_t {
    viewer_default,
    off,
    xy_grid,
    xz_grid,
    yz_grid,
};
bool set_reference_grid(reference_grid value);

// Request a render PNG from the viewer. The default waits for full accumulation;
// pass false to save at the first sample.
bool request_capture(bool full_accumulation = true);
```

- ローカル viewer (`127.0.0.1:47909`) への既定接続は、最初の送信系 API 呼び出し時に暗黙接続されます。別ホストへ接続する場合は `connect()` を明示します。\
The default connection to the local viewer (`127.0.0.1:47909`) is established implicitly on the first sending API call. Call `connect()` explicitly to connect to another host.
- `push_layer("name")` / `push_layerf("mesh_%u", mesh_id)` / `pop_layer()` で primitive をレイヤーに分類できます。`push_layerf()` は `printf` と同じ形式指定でレイヤー名を組み立てます。入れ子は親子関係を持って viewer の Layers UI に表示されます。\
`push_layer("name")` / `push_layerf("mesh_%u", mesh_id)` / `pop_layer()` classify primitives into layers. `push_layerf()` formats the layer name with `printf`-style arguments. Nested layers form parent-child relationships in the viewer's Layers UI.
- レイヤー名は空文字と `/` を含む名前を受け付けず、1階層あたり63 byteまでです。\
Layer names cannot be empty or contain `/`, and are limited to 63 bytes per level.
- `set_reference_grid()` は viewer に表示する座標軸と基準グリッドの平面の変更を要求します。`off` は非表示、
  `xy_grid` / `xz_grid` / `yz_grid` は対応する平面を表示します。指定は永続的な client 設定ではないため、
  呼び出し後も viewer の Display UI にある `XYZ Grid` 設定で自由に変更できます。`viewer_default` は変更しません。\
`set_reference_grid()` requests a change to the coordinate axes and reference-grid plane shown by the viewer. `off`
hides them, while `xy_grid`, `xz_grid`, and `yz_grid` select the corresponding plane. It is not a persistent client
setting, so the viewer's `XYZ Grid` setting in the Display UI remains freely editable after the call.
`viewer_default` makes no change.
- `request_capture()` は要求受信時点の viewer が保持している scene / 視点を対象にします。scene の snapshot は保持しないため、要求時に表示したい scene を安定した状態で送ることは client 側の責任です。フル accumulation 中に viewer のカメラを操作すると accumulation がリセットされ、保存までの時間が延びることがあります。`request_capture(false)` は accumulation を待たず、その時点の 1 spp 画像を保存します。\
`request_capture()` captures the scene and view currently held by the viewer when the request is received. It does not retain or copy a scene snapshot, so the client is responsible for keeping the submitted scene stable when requesting a capture. Moving the viewer camera during full accumulation resets accumulation and may delay the save; `request_capture(false)` saves the current 1-spp image without waiting for full accumulation.
- 保存要求があった場合だけ、Windows/Linux では viewer 実行ファイルと同じディレクトリに、macOS では `~/Library/Application Support/rtvdb/sessions/` にセッションディレクトリを作成します。\
Capture session directories are created only after a request: beside the viewer executable on Windows/Linux, and under `~/Library/Application Support/rtvdb/sessions/` on macOS.

## Samples

- `rtvdb_basic_client`: Minimal connection check
- `rtvdb_bt2020_volume_client`: Color-gamut volume sample using points and line segments
- `rtvdb_obj_stream_client`: Sends an OBJ incrementally
  - Add `--wireframe` to send unique OBJ face-boundary edges as line primitives without triangles.

```powershell
.\build\Debug\rtvdb_obj_stream_client.exe --mesh .\mesh.obj --wireframe --batch 512 --sleep-ms 0
```

Wireframe extraction treats edges as undirected geometry edges, removes shared or duplicate edges, and does not
include triangulation diagonals from polygon faces. With `--limit-triangles`, only faces covered by the triangle limit
contribute edges; `--color-rgba` also controls the wireframe color. Wireframe line radius is automatically set to 0.1%
of the mesh AABB's longest extent, or can be overridden with `--line-radius <value>`.

## Remote Viewer Connection

viewer は既定では localhost のみ待ち受けます。別マシンの client から接続させる場合だけ、viewer 起動時に待受アドレスを明示します。\
By default, the viewer listens only on localhost. Specify a listening address when starting the viewer only if clients on another machine need to connect.

```powershell
.\build\Debug\rtvdb_viewer.exe --listen-host 0.0.0.0 --listen-port 47909
```

client 側では `config.host` に viewer の IP を指定します。\
On the client side, set the viewer's IP address in `config.host`.

```cpp
rtvdb::config cfg{};
cfg.host = "192.168.1.10";
cfg.port = 47909;
if (!rtvdb::connect(&cfg, "remote_app")) {
    return 1;
}
```

## Supported Platforms

client library と viewer は対応範囲が異なります。\
The supported platforms differ between the client library and the viewer.

| Platform | Client library | Viewer backend | Current status |
| --- | --- | --- | --- |
| Windows 10/11 | Supported | D3D12 DXR | Primary target and default configuration |
| Windows 10/11 | Supported | Vulkan RT | Opt-in and experimental |
| Linux | Supported | Vulkan RT | Experimental and not hardware-validated |
| FreeBSD | Supported | Vulkan RT | Experimental and not hardware-validated |
| macOS | Supported | Metal RT | Experimental |

- client library は Windows では WinSock、Linux / FreeBSD / macOS では POSIX socket を使用します。\
The client library uses WinSock on Windows and POSIX sockets on Linux, FreeBSD, and macOS.
- Windows DXR viewer の実行には、DXR 対応GPUと対応ドライバーが必要です。\
Running the Windows DXR viewer requires a DXR-capable GPU and a compatible driver.
- Vulkan RT viewer の実行には、Vulkan 1.3とray tracing extensionに対応したGPUおよびドライバーが必要です。\
Running a Vulkan RT viewer requires a GPU and driver supporting Vulkan 1.3 and the ray tracing extensions.
- 非RT環境向けfallback backendは提供しません。\
No fallback backend is provided for non-RT environments.
- macOS Metal RT backendはnative presentに対応していますが、実験的な位置付けです。\
The macOS Metal RT backend supports native presentation but remains experimental.
- CIによる全プラットフォームの継続検証はまだ整備されていません。\
Continuous CI validation across all platforms is not yet in place.

実行時メモ / Runtime notes:

- Windows 配布で Release viewer をそのまま使う場合、MSVC DLL runtime を解決できる環境が必要です。\
When using the distributed Release viewer on Windows, the environment must be able to resolve the MSVC DLL runtime.
- Windows build で Vulkan RT backend を有効にした viewer は、実行時に Vulkan loader (`vulkan-1.dll`) を解決できる環境が必要です。\
A viewer built with the Vulkan RT backend enabled on Windows requires the Vulkan loader (`vulkan-1.dll`) at runtime.
- Windows 既定構成は DXR backend のみで、Vulkan RT backend は configure 時に明示的に有効化します。\
The default Windows configuration contains only the DXR backend; enable the Vulkan RT backend explicitly at configure time.

## Build

ここからはviewerやサンプルclientをソースからビルドする場合の情報です。

共通要件 / Common requirements:

- CMake 3.24以上
- C++20対応コンパイラー
- Git submoduleで取得する依存ライブラリ

backend別の追加要件

| backend | ビルド時に必要な環境 |
| --- | --- |
| Windows D3D12 DXR | C++20対応MSVC、`dxc.exe`を含むWindows SDK |
| Windows / Linux / FreeBSD Vulkan RT | Vulkan SDKまたは同等のVulkan headers / loader、SPIR-V出力対応`dxc` |
| macOS Metal RT | Objective-C++対応Clang、`metal`と`metallib`を含むMetal Toolchain |

依存の取得

```powershell
git submodule update --init --recursive
```

### SDL Vulkan 低レイテンシーパッチ / SDL Vulkan low-latency patch

Vulkan RT をビルドする場合、CMake configure は submodule の初期化または更新後に
rtvdb 管理の SDL patch series を自動適用します。\
For Vulkan RT builds, CMake configure automatically applies the rtvdb-managed SDL patch series
after submodules have been initialized or updated.

```powershell
cmake -S . -B build -DRTVDB_ENABLE_VULKAN_RT=ON
```

このパッチは、表示レイテンシーを抑えるため SDL Vulkan renderer が要求する swapchain image 数を
surface の最小数にします。CMake は未適用の patch だけを適用し、適用済みの場合は何も変更しません。\
The patch reduces the SDL Vulkan renderer's requested swapchain image count to the surface
minimum for lower presentation latency. CMake applies only patches that have not already been
applied and leaves an already patched submodule unchanged.

SDL 更新後は次回の Vulkan RT configure 時に再適用されます。適用に失敗した場合は、更新された SDL に
patch を追従させてから再configureしてください。適用・更新時の詳細およびライセンス上の扱いは
[`third_party/patches/SDL/README.md`](third_party/patches/SDL/README.md) を参照してください。\
After an SDL update, it is reapplied by the next Vulkan RT configure. If it fails, update the
patch for the new SDL source before reconfiguring. See
[`third_party/patches/SDL/README.md`](third_party/patches/SDL/README.md) for update and license details.

Windows既定構成のビルド

```powershell
cmake -S . -B build
cmake --build build --config Debug
```

WindowsでVulkan RT backendを有効にする場合は、configure時に
`-DRTVDB_ENABLE_VULKAN_RT=ON`を指定します。

主な生成物 / Main build outputs:
- `build\Debug\rtvdb_viewer.exe`
- `build\Debug\rtvdb_basic_client.exe`
- `build\Debug\rtvdb_obj_stream_client.exe`
- `build\Debug\rtvdb_bt2020_volume_client.exe`

----
2026 [@Shocker_0x15](https://twitter.com/Shocker_0x15), [@bsky.rayspace.xyz](https://bsky.app/profile/bsky.rayspace.xyz)
