import { auth } from "@/lib/auth";
import { Loader2 } from "lucide-react";
import { headers } from "next/headers";
import { redirect } from "next/navigation";
import { Suspense } from "react";
import { SongPanel } from "./components/song-panel";
import TrackListFetcher from "./components/track-list-fether";

export default async function Page() {
  return (
    <div className="flex h-full flex-col lg:flex-row">
      <SongPanel />
      <Suspense
        fallback={
          <div className="flex h-full w-full items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin" />
          </div>
        }
      >
        <TrackListFetcher />
      </Suspense>
    </div>
  );
}
