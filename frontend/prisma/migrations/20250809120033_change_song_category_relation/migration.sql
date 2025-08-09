-- DropForeignKey
ALTER TABLE "public"."Song" DROP CONSTRAINT "Song_categoryId_fkey";

-- CreateTable
CREATE TABLE "public"."_CategoryToSong" (
    "A" TEXT NOT NULL,
    "B" TEXT NOT NULL,

    CONSTRAINT "_CategoryToSong_AB_pkey" PRIMARY KEY ("A","B")
);

-- CreateIndex
CREATE INDEX "_CategoryToSong_B_index" ON "public"."_CategoryToSong"("B");

-- AddForeignKey
ALTER TABLE "public"."_CategoryToSong" ADD CONSTRAINT "_CategoryToSong_A_fkey" FOREIGN KEY ("A") REFERENCES "public"."Category"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."_CategoryToSong" ADD CONSTRAINT "_CategoryToSong_B_fkey" FOREIGN KEY ("B") REFERENCES "public"."Song"("id") ON DELETE CASCADE ON UPDATE CASCADE;
