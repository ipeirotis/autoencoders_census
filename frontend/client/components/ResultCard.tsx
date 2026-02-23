import React from "react";
import { Copy, Check, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ResultCardProps {
  type: "success" | "error";
  message: string;
  onCopy?: () => void;
  copied?: boolean;
}

export const ResultCard: React.FC<ResultCardProps> = ({
  type,
  message,
  onCopy,
  copied = false,
}) => {
  return (
    <div
      data-testid="result-card"
      className={cn(
        "rounded-2xl p-6 border-2 space-y-4",
        type === "success"
          ? "bg-green-50 border-green-200"
          : "bg-red-50 border-red-200"
      )}
      role="status"
      aria-live="polite"
    >
      <div className="flex items-start gap-3">
        {type === "success" ? (
          <div className="mt-1">
            <span className="inline-block bg-green-100 text-green-700 px-3 py-1 rounded-full text-xs font-semibold">
              Uploaded
            </span>
          </div>
        ) : (
          <AlertCircle className="w-5 h-5 text-red-500 mt-1 flex-shrink-0" />
        )}
      </div>

      <div>
        <p
          className={cn(
            "text-sm",
            type === "success" ? "text-gray-700" : "text-red-700"
          )}
        >
          {message}
        </p>
      </div>

      {onCopy && (
        <Button
          onClick={onCopy}
          variant="outline"
          size="sm"
          className={cn(
            "gap-2",
            type === "success"
              ? "border-green-300 text-green-700 hover:bg-green-100"
              : "border-red-300 text-red-700 hover:bg-red-100"
          )}
        >
          {copied ? (
            <>
              <Check className="w-4 h-4" />
              Copied!
            </>
          ) : (
            <>
              <Copy className="w-4 h-4" />
              Copy
            </>
          )}
        </Button>
      )}
    </div>
  );
};
