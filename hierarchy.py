#!/usr/bin/env python3
"""
Affiche la hiérarchie d'un dossier sous forme d'arborescence.
Usage : python hierarchy.py [chemin] [options]
"""
import re
import os
import argparse


def get_tree(
    path: str,
    prefix: str = "",
    max_depth: int = None,
    current_depth: int = 0,
    show_hidden: bool = False,
    dirs_only: bool = False,
    exclude_exts: set = None,
) -> list[str]:
    """Construit récursivement l'arborescence d'un dossier."""

    if max_depth is not None and current_depth >= max_depth:
        return []

    try:
        entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        return [prefix + "└── [Permission refusée]"]

    # Filtres
    if not show_hidden:
        entries = [e for e in entries if not e.name.startswith(".")]
    if dirs_only:
        entries = [e for e in entries if e.is_dir()]
    if exclude_exts:
        entries = [
            e for e in entries
            if e.is_dir() or os.path.splitext(e.name)[1].lower() not in exclude_exts
        ]

    lines = []
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "

        # Icône selon le type
        if entry.is_dir():
            icon = "📁 "
        elif entry.is_file():
            ext = os.path.splitext(entry.name)[1].lower()
            icons = {
                ".py": "🐍 ", ".js": "📜 ", ".ts": "📜 ",
                ".html": "🌐 ", ".css": "🎨 ",
                ".json": "📋 ", ".yaml": "📋 ", ".yml": "📋 ", ".toml": "📋 ",
                ".md": "📝 ", ".txt": "📝 ", ".rst": "📝 ",
                ".pdf": "📄 ", ".docx": "📄 ", ".xlsx": "📄 ",
                ".png": "🖼️ ", ".jpg": "🖼️ ", ".jpeg": "🖼️ ", ".svg": "🖼️ ", ".gif": "🖼️ ",
                ".mp4": "🎬 ", ".mp3": "🎵 ",
                ".zip": "📦 ", ".tar": "📦 ", ".gz": "📦 ",
                ".sh": "⚙️ ", ".bat": "⚙️ ",
                ".env": "🔒 ", ".gitignore": "🔒 ",
                ".sql": "🗄️ ",
            }
            icon = icons.get(ext, "📄 ")
        else:
            icon = "🔗 "

        # Taille si fichier
        size_str = ""
        if entry.is_file():
            try:
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"  [{size} B]"
                elif size < 1024 ** 2:
                    size_str = f"  [{size / 1024:.1f} KB]"
                elif size < 1024 ** 3:
                    size_str = f"  [{size / 1024**2:.1f} MB]"
                else:
                    size_str = f"  [{size / 1024**3:.1f} GB]"
            except OSError:
                pass

        lines.append(f"{prefix}{connector}{icon}{entry.name}{size_str}")

        if entry.is_dir():
            sub_lines = get_tree(
                entry.path,
                prefix=prefix + extension,
                max_depth=max_depth,
                current_depth=current_depth + 1,
                show_hidden=show_hidden,
                dirs_only=dirs_only,
                exclude_exts=exclude_exts,
            )
            lines.extend(sub_lines)

    return lines


def count_items(path: str, show_hidden: bool = False, exclude_exts: set = None) -> tuple[int, int]:
    """Compte le nombre total de fichiers et dossiers."""
    n_files, n_dirs = 0, 0
    for root, dirs, files in os.walk(path):
        if not show_hidden:
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            files = [f for f in files if not f.startswith(".")]
        if exclude_exts:
            files = [f for f in files if os.path.splitext(f)[1].lower() not in exclude_exts]
        n_dirs += len(dirs)
        n_files += len(files)
    return n_files, n_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Affiche la hiérarchie d'un dossier en excluant les fichiers DICOM.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Arguments optionnels
    parser.add_argument("-d", "--depth", type=int, default=None, metavar="N",
                        help="Profondeur maximale d'exploration")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Inclure les fichiers et dossiers cachés")
    parser.add_argument("--dirs-only", action="store_true",
                        help="Afficher uniquement les dossiers")
    parser.add_argument("-e", "--exclude", nargs="+", default=[], metavar="EXT",
                        help="Extensions supplémentaires à exclure")
    parser.add_argument("--no-icons", action="store_true",
                        help="Ne pas afficher les icônes")

    args = parser.parse_args()

    # --- PARTIE INTERACTIVE ---
    print("--- Générateur de Hiérarchie (Exclusion DICOM active) ---")
    user_path = input("Entrez le chemin du dossier à analyser : ").strip()
    
    if not user_path:
        user_path = "."

    # --- FILTRE DICOM ---
    # On ajoute d'office .dcm à la liste des exclusions
    exclude_exts = {".dcm"} 
    for ext in args.exclude:
        ext = ext.lower()
        if not ext.startswith("."):
            ext = "." + ext
        exclude_exts.add(ext)

    # Vérification du chemin
    path = os.path.abspath(user_path)
    if not os.path.exists(path):
        print(f"❌ Erreur : le chemin '{path}' n'existe pas.")
        return
    if not os.path.isdir(path):
        print(f"❌ Erreur : '{path}' n'est pas un dossier.")
        return

    # Extraction du nom du dossier
    folder_name = os.path.basename(path)
    if not folder_name:
        folder_name = path

    # Construction de l'arbre
    lines = get_tree(
        path,
        max_depth=args.depth,
        show_hidden=args.all,
        dirs_only=args.dirs_only,
        exclude_exts=exclude_exts,
    )

    # Gestion des icônes
    if args.no_icons:
        lines = [re.sub(r"[^\x00-\x7F]+\s?", "", line) for line in lines]

    depth_str = f" (profondeur max : {args.depth})" if args.depth else ""
    exclude_str = f" (exclus : {', '.join(sorted(exclude_exts))})"
    
    header = f"📂 {folder_name}{depth_str}{exclude_str}\n"
    body = "\n".join(lines)

    # Statistiques
    n_files, n_dirs = count_items(path, show_hidden=args.all, exclude_exts=exclude_exts)
    summary = f"\n{n_dirs} dossier(s), {n_files} fichier(s)"

    full_output = header + body + summary
    
    # 1. Affichage console
    print("\n" + full_output)

    # 2. Enregistrement dans P01.txt
    output_filename = "P01.txt"
    clean = lambda s: re.sub(r"[^\x00-\x7F]+\s?", "", s)
    clean_output = clean(header) + "\n".join(clean(l) for l in lines) + summary
    
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(clean_output)
        print(f"\n✅ Structure (sans DICOM) enregistrée dans : {output_filename}")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        
        
if __name__ == "__main__":
    main()