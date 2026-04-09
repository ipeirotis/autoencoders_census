/**
 * File upload drag-and-drop component
 *
 * This component allows users to upload CSV files by dragging and dropping them
 * into a designated area or by clicking to open a file dialog.
 */
import React, { useState, useRef } from "react";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";
import { fileTypeFromBuffer } from 'file-type';
import { useToast } from "@/hooks/use-toast";

interface DropzoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

/**
 * Validate file for CSV upload
 * Checks both extension and magic bytes to prevent binary files disguised as CSV
 */
async function validateFile(file: File): Promise<{ valid: boolean; error?: string }> {
  // Check extension (case-insensitive — Windows commonly produces .CSV)
  if (!file.name.toLowerCase().endsWith('.csv')) {
    return { valid: false, error: 'Only CSV files are allowed' };
  }

  // Check magic bytes (prevents .exe renamed to .csv).
  // file-type only inspects the first few bytes, so read a small slice
  // instead of buffering the entire file (up to 50 MB) into memory.
  const MAGIC_BYTE_SAMPLE_SIZE = 4100; // file-type recommended minimum
  const sample = await file.slice(0, MAGIC_BYTE_SAMPLE_SIZE).arrayBuffer();
  const type = await fileTypeFromBuffer(new Uint8Array(sample));

  // CSV files are text/plain or text/csv, may not have magic bytes
  // If fileTypeFromBuffer returns undefined (text file), that's OK for CSV
  // If it returns a type, ensure it's NOT a binary format
  if (type && !['text/csv', 'text/plain'].includes(type.mime)) {
    return { valid: false, error: `Invalid file type: ${type.mime}. Only CSV files allowed.` };
  }

  return { valid: true };
}

export const Dropzone: React.FC<DropzoneProps> = ({
  onFileSelect,
  disabled = false,
}) => {
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();

    if (disabled) return;

    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();

    if (disabled) return;

    setIsDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];

      // Validate file with shared validation function
      const validation = await validateFile(file);
      if (!validation.valid) {
        toast({
          title: 'Invalid File',
          description: validation.error,
          variant: 'destructive'
        });
        return;
      }

      onFileSelect(file);
    }
  };

  const handleInputChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files;
    if (files && files.length > 0) {
      const file = files[0];

      // Validate file with shared validation function
      const validation = await validateFile(file);
      if (!validation.valid) {
        toast({
          title: 'Invalid File',
          description: validation.error,
          variant: 'destructive'
        });
        return;
      }

      onFileSelect(file);
    }
  };

  const handleClick = () => {
    if (!disabled) {
      fileInputRef.current?.click();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;

    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInputRef.current?.click();
    }
  };

  return (
    <div
      data-testid="upload-dropzone"
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label="Upload CSV file - drop here or click to browse"
      aria-disabled={disabled}
      className={cn(
        "relative border-2 border-dashed rounded-2xl p-12 text-center transition-all cursor-pointer",
        "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2",
        {
          "border-blue-500 bg-blue-50": isDragActive,
          "border-gray-300 hover:bg-gray-50": !isDragActive && !disabled,
          "border-gray-200 bg-gray-50 cursor-not-allowed opacity-60": disabled,
        }
      )}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,text/csv"
        onChange={handleInputChange}
        disabled={disabled}
        className="hidden"
        aria-hidden="true"
      />

      <div className="flex flex-col items-center gap-3">
        <Upload
          className={cn("w-8 h-8", {
            "text-blue-500": isDragActive,
            "text-gray-400": !isDragActive,
          })}
        />
        <p className="text-gray-800 font-medium">Drop CSV here or click to browse</p>
      </div>
    </div>
  );
};
