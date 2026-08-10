# SDL local patches

`third_party/SDL` tracks the unmodified upstream SDL commit. DXR or Vulkan RT CMake configure
applies this patch series after the submodule has been initialized or updated:

```powershell
cmake -S . -B build -DRTVDB_ENABLE_VULKAN_RT=ON
cmake -S . -B build-dxr -DRTVDB_ENABLE_D3D12_DXR=ON -DRTVDB_ENABLE_VULKAN_RT=OFF
```

Only patches not already applied are changed. The script can also be run explicitly when needed:

```powershell
cmake -DRTVDB_ENABLE_VULKAN_RT=ON -P cmake/apply_sdl_patches.cmake
cmake -DRTVDB_ENABLE_D3D12_DXR=ON -P cmake/apply_sdl_patches.cmake
```

The script selects Vulkan and D3D12 patches independently from their filenames, so a DXR-only
configure does not depend on the Vulkan patch series and vice versa.

After an SDL update, CMake applies the relevant series again at the next DXR or Vulkan RT configure. If
`git apply --3way` cannot apply a patch cleanly, resolve it against the updated SDL source and
update the patch file. For every updated patch, retain its tested upstream tag and commit in the
patch header, then rebuild the Release viewer and run the corresponding native-render validation.

`0001-rtvdb-low-latency-vulkan-swapchain.patch` makes the SDL Vulkan renderer's queue-depth
macro externally configurable. rtvdb supplies `SDL_VULKAN_FRAME_QUEUE_DEPTH=0` for Vulkan RT,
so SDL requests only the surface minimum number of swapchain images.

`0002-rtvdb-vulkan-present-current-frame-wait.patch` makes the Vulkan renderer wait for the
current composition submission at present. This matches the SDL D3D12 renderer's queue-drain
boundary when rtvdb submits RT work on the same graphics queue. It was written against SDL
`release-3.4.10` (`8e37db5e797b6167f3a00d697d816a684bd259c7`). The patch must remain after
`vkQueuePresentKHR()` and before `currentCommandBufferIndex` advances; the header records the
reasoning and the required ordering for future rebases.

`0003-rtvdb-d3d12-external-resource-state.patch` lets a wrapped D3D12 texture declare the
resource state SDL should track. rtvdb creates its shared native target in
`PIXEL_SHADER_RESOURCE`, so the DXR copy and SDL sampling paths share one explicit state contract
without duplicate barriers. Validate it with the D3D12 debug layer on initial creation, resize,
and `--recreate-native-target-each-frame`.

SDL is distributed under the zlib license. Keep `third_party/SDL/LICENSE.txt` intact, retain
this patch as the plain marking of the altered source, and include the installed
`licenses/SDL/LICENSE.txt` when distributing an rtvdb build that contains SDL.
