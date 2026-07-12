#import <AppKit/AppKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include "viewer_shell/shell.h"

#include <filesystem>

namespace rtvdb::viewer_shell {

namespace {

NSString* nsstring_from_wstring(const std::wstring &text) {
    const std::string utf8 = std::filesystem::path(text).string();
    if (utf8.empty()) {
        return @"";
    }
    return [[NSString alloc] initWithUTF8String:utf8.c_str()];
}

} // namespace

bool request_png_save_path(
    const std::wstring &suggested,
    png_save_path_callback callback,
    void* user_data)
{
    if (callback == nullptr) {
        return false;
    }

    @autoreleasepool {
        NSSavePanel* panel = [NSSavePanel savePanel];
        [panel setCanCreateDirectories:YES];
        UTType* png_type = [UTType typeWithFilenameExtension:@"png"];
        if (png_type != nil) {
            [panel setAllowedContentTypes:@[png_type]];
        }
        [panel setExtensionHidden:NO];
        [panel setNameFieldStringValue:nsstring_from_wstring(std::filesystem::path(suggested).filename().wstring())];

        NSWindow* owner = nil;
        const native_window_handle window = native_window();
        if (window.kind == native_window_kind::cocoa_nswindow) {
            owner = static_cast<NSWindow*>(window.value);
        }
        if (owner == nil) {
            return false;
        }

        [NSApp activateIgnoringOtherApps:YES];
        [owner makeKeyAndOrderFront:nil];
        dispatch_async(dispatch_get_main_queue(), ^{
            [panel beginSheetModalForWindow:owner completionHandler:^(NSModalResponse response) {
                std::wstring path;
                if (response == NSModalResponseOK) {
                    NSURL* url = [panel URL];
                    if (url != nil && [url isFileURL]) {
                        NSString* path_string = [[url path] precomposedStringWithCanonicalMapping];
                        const char* utf8 = path_string != nil ? [path_string UTF8String] : nullptr;
                        if (utf8 != nullptr) {
                            path = std::filesystem::path(utf8).wstring();
                            if (std::filesystem::path(path).extension().empty()) {
                                path += L".png";
                            }
                        }
                    }
                }
                callback(!path.empty(), path, user_data);
            }];
        });
        return true;
    }
}

} // namespace rtvdb::viewer_shell
