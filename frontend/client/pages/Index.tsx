import React, { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { Dropzone } from "@/components/Dropzone";
import { PreviewTable } from "@/components/PreviewTable";
import { ResultCard } from "@/components/ResultCard";
import { parseCSVFile, type CSVParseResult } from "@/utils/csv-parser";
import { uploadCsv, type UploadResponse } from "@/utils/api";
import { cn } from "@/lib/utils";

interface UploadState {
  file: File | null;
  preview: CSVParseResult | null;
  error: string | null;
  response: UploadResponse | null;
  copied: boolean;
}

const INITIAL_STATE: UploadState = {
  file: null,
  preview: null,
  error: null,
  response: null,
  copied: false,
};

export default function Index() {
  const [state, setState] = useState<UploadState>(INITIAL_STATE);
  const [skipRows, setSkipRows] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  // User uploads CSV file
  const handleFileSelect = useCallback(
    async (file: File) => {
      // Validate file size (warn if > 50MB)
      const MAX_SIZE = 50 * 1024 * 1024;
      if (file.size > MAX_SIZE) {
        if (
          !window.confirm(
            `File is ${(file.size / 1024 / 1024).toFixed(1)}MB (max recommended: 50MB). Continue?`
          )
        ) {
          return;
        }
      }

      setState((prev) => ({ ...prev, file, error: null }));

      try {
        // Parse CSV for preview
        const preview = await parseCSVFile(file, skipRows);
        setState((prev) => ({ ...prev, preview, error: null }));
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Failed to parse CSV";
        setState((prev) => ({ ...prev, error: errorMessage, preview: null }));
        toast({
          variant: "destructive",
          title: "CSV Parse Error",
          description: errorMessage,
        });
      }
    },
    [skipRows, toast]
  );
  // User clicks upload button
  const handleUpload = useCallback(async () => {
    if (!state.file) return;

    setIsLoading(true);
    try {
      const response = await uploadCsv(state.file, skipRows);

      // Save to localStorage
      localStorage.setItem(
        "last_upload",
        JSON.stringify({
          dataset_id: response.dataset_id,
          filename: state.file.name,
          uploadedAt: new Date().toISOString(),
        })
      );

      setState((prev) => ({
        ...prev,
        response,
        error: null,
      }));

      toast({
        title: "Success",
        description: "CSV uploaded successfully",
      });
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Upload failed";
      setState((prev) => ({ ...prev, error: errorMessage }));
      toast({
        variant: "destructive",
        title: "Upload Error",
        description: errorMessage,
      });
    } finally {
      setIsLoading(false);
    }
  }, [state.file, skipRows, toast]);

  const handleReset = useCallback(() => {
    setState(INITIAL_STATE);
    setSkipRows(0);
  }, []);

  const handleCopyDatasetId = useCallback(() => {
    if (state.response?.dataset_id) {
      navigator.clipboard.writeText(state.response.dataset_id);
      setState((prev) => ({ ...prev, copied: true }));
      setTimeout(() => {
        setState((prev) => ({ ...prev, copied: false }));
      }, 2000);
    }
  }, [state.response?.dataset_id]);

  const hasFile = !!state.file;
  const hasPreview = !!state.preview;
  const hasResponse = !!state.response;
  const hasError = !!state.error;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900">CSV Upload</h1>
          <p className="mt-2 text-gray-600">
            Upload your CSV file to process and analyze
          </p>
        </div>

        {/* Main Content */}
        <div className="space-y-6">
          {/* Dropzone Card */}
          <div className="bg-white rounded-2xl shadow p-6">
            <Dropzone onFileSelect={handleFileSelect} disabled={isLoading} />
          </div>

          {/* Skip Rows Input */}
          {hasFile && (
            <div className="bg-white rounded-2xl shadow p-6">
              <div className="space-y-3">
                <Label htmlFor="skip-rows" className="text-sm font-medium text-gray-900">
                  Skip rows from beginning
                </Label>
                <Input
                  id="skip-rows"
                  type="number"
                  min="0"
                  max="1000"
                  value={skipRows}
                  onChange={(e) => {
                    const newSkipRows = Math.max(0, parseInt(e.target.value) || 0);
                    setSkipRows(newSkipRows);
                    if (state.file) {
                      handleFileSelect(state.file);
                    }
                  }}
                  disabled={isLoading}
                  placeholder="0"
                  className="w-full"
                />
                <p className="text-xs text-gray-500">
                  Useful for skipping header rows or metadata. Leave at 0 to include all rows.
                </p>
              </div>
            </div>
          )}

          {/* Preview Card */}
          {hasPreview && (
            <div className="bg-white rounded-2xl shadow p-6">
              <div className="space-y-4">
                <div>
                  <p className="text-sm font-medium text-gray-900">
                    File: {state.file?.name}
                  </p>
                  <p className="text-sm text-gray-600">
                    Size: {(state.file?.size || 0) > 1024 * 1000
                      ? ((state.file?.size || 0) / 1024 / 1024).toFixed(1) + ' MB'
                      : ((state.file?.size || 0) / 1024).toFixed(1) + ' KB'}
                  </p>
                </div>
                <PreviewTable
                  rows={state.preview.rows}
                  headers={state.preview.headers}
                  totalRows={state.preview.totalRows}
                />
              </div>
            </div>
          )}

          {/* Error Alert */}
          {hasError && !hasResponse && (
            <ResultCard
              type="error"
              message={state.error}
              onCopy={
                state.error
                  ? () => {
                      navigator.clipboard.writeText(state.error!);
                      setState((prev) => ({ ...prev, copied: true }));
                      setTimeout(() => {
                        setState((prev) => ({ ...prev, copied: false }));
                      }, 2000);
                    }
                  : undefined
              }
              copied={state.copied}
            />
          )}

          {/* Action Buttons */}
          {hasFile && !hasResponse && (
            <div className="flex gap-3">
              <Button
                data-testid="upload-button"
                onClick={handleUpload}
                disabled={!hasFile || isLoading}
                size="lg"
                className={cn(
                  "flex-1",
                  isLoading && "opacity-75"
                )}
              >
                {isLoading ? (
                  <>
                    <svg
                      className="animate-spin -ml-1 mr-2 h-5 w-5"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    Uploading...
                  </>
                ) : (
                  "Upload"
                )}
              </Button>
              <Button
                onClick={handleReset}
                variant="outline"
                size="lg"
              >
                Reset
              </Button>
            </div>
          )}

          {/* Success Card */}
          {hasResponse && (
            <div className="space-y-4">
              <ResultCard
                type="success"
                message={`Dataset ID: ${state.response.dataset_id}`}
                onCopy={handleCopyDatasetId}
                copied={state.copied}
              />
              <Button
                onClick={handleReset}
                variant="outline"
                size="lg"
                className="w-full"
              >
                Upload Another File
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
