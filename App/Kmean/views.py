from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import requests
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.cluster import KMeans
import json


# Create your views here.
@csrf_exempt
def kmean(request):
    if request.method == "GET":
        # Xử lý GET request
        return HttpResponse("This is a GET request")
    elif request.method == "POST":
        try:
            # Dữ liệu bạn muốn truyền trong request body
            data = {"query": {"match_all": {}}, "size": 999}

            # Gọi API và truyền dữ liệu trong request body
            response = requests.get(
                "http://localhost:9200/userslist/_search", json=data
            )

            # Kiểm tra mã trạng thái của response
            if response.status_code == 200:
                # Xử lý dữ liệu trả về nếu cần thiết
                api_response = response.json()

                data = create_dataframe(api_response.get("hits", {}).get("hits", []))

                data["label"] = ""

                for index, row in data.iterrows():
                    label = ""
                    if row["age"] <= 0.15:
                        label += "age_low, "
                    elif row["age"] <= 0.4:
                        label += "age_average_low, "
                    elif row["age"] <= 0.6:
                        label += "age_average_high, "
                    elif row["age"] <= 0.75:
                        label += "age_high, "
                    else:
                        label += "age_super_high, "

                    if row["annual_income"] <= 0.15:
                        label += "annual_income_low, "
                    elif row["annual_income"] <= 0.4:
                        label += "annual_income_average_low, "
                    elif row["annual_income"] <= 0.6:
                        label += "annual_income_average_high, "
                    elif row["annual_income"] <= 0.75:
                        label += "annual_income_high, "
                    else:
                        label += "annual_super_high, "

                    if row["spending_score"] <= 0.15:
                        label += "spending_score_low"
                    elif row["spending_score"] <= 0.4:
                        label += "spending_score_average_low"
                    elif row["spending_score"] <= 0.6:
                        label += "spending_score_average_high"
                    elif row["spending_score"] <= 0.75:
                        label += "spending_score_high"
                    else:
                        label += "spending_super_high"

                    data.at[index, "name"] = label.strip()

                df_json = data.to_dict(orient="records")

                # Trả về dữ liệu từ API
                return JsonResponse({"metadata": df_json}, safe=False)
            else:
                # Nếu không thành công, trả về thông báo lỗi
                return JsonResponse({"error": "Failed to call API"}, status=500)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    elif request.method == "PUT":
        # Xử lý PUT request
        data = request.body  # PUT data thường nằm trong request.body
        return JsonResponse({"message": "This is a PUT request", "data": data})
    elif request.method == "DELETE":
        # Xử lý DELETE request
        return HttpResponse("This is a DELETE request")
    else:
        return HttpResponse("Method not allowed", status=405)


@csrf_exempt
def findCountAge(request):
    if request.method == "GET":
        # Xử lý GET request
        return HttpResponse("This is a GET request")
    elif request.method == "POST":
        try:
            dataQuery = [
                {
                    "query": {
                        "range": {
                            "day_of_bight": {"gte": "2006-01-01", "lte": "2019-01-01"}
                        }
                    }
                },
                {
                    "query": {
                        "range": {
                            "day_of_bight": {"gte": "1986-01-01", "lte": "2006-01-01"}
                        }
                    }
                },
                {
                    "query": {
                        "range": {
                            "day_of_bight": {"gte": "1966-01-01", "lte": "1986-01-01"}
                        }
                    }
                },
                {"query": {"range": {"day_of_bight": {"lte": "1966-01-01"}}}},
            ]

            label = ["Under 18 years of age", "Age 18-45", "Age 45-65", "Age over 65"]

            data = []

            for index, value in enumerate(dataQuery):
                response = requests.get(
                    "http://localhost:9200/userslist/_search", json=value
                )

                api_response = response.json()
                if response.status_code == 200:
                    data.append(
                        {
                            "label": label[index],
                            "value": api_response.get("hits", {})
                            .get("total", {})
                            .get("value"),
                        }
                    )
                else:
                    # Nếu không thành công, trả về thông báo lỗi
                    return JsonResponse({"error": "Failed to call API"}, status=500)
            print(data)
            return JsonResponse({"metadata": data})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    elif request.method == "PUT":
        # Xử lý PUT request
        data = request.body  # PUT data thường nằm trong request.body
        return JsonResponse({"message": "This is a PUT request", "data": data})
    elif request.method == "DELETE":
        # Xử lý DELETE request
        return HttpResponse("This is a DELETE request")
    else:
        return HttpResponse("Method not allowed", status=405)


def create_dataframe(data):
    # Khởi tạo một danh sách rỗng để lưu trữ các dòng dữ liệu
    rows = []

    # Lặp qua từng phần tử trong danh sách data
    for item in data:
        # Lấy các giá trị trong trường "_source"
        source = item.get("_source", {})

        # Tính tuổi từ ngày sinh
        day_of_birth = source.get("day_of_bight", "")
        age = calculate_age(day_of_birth)

        # Tạo một dictionary mới để lưu trữ các giá trị từ "_source" và tuổi
        row = {
            "usr_id": source.get("usr_id", ""),
            "day_of_bight": day_of_birth,
            "age": age,
            "gender": source.get("gender", ""),
            "annual_income": source.get("annual_income", 0),
            "spending_score": source.get("spending_score", 0),
        }

        # Thêm dictionary này vào danh sách rows
        rows.append(row)

    # Tạo DataFrame từ danh sách các dòng dữ liệu
    df = pd.DataFrame(rows)
    df["gender"].replace({"F": 0, "M": 1}, inplace=True)

    # Danh sách các cột cần chuyển đổi
    columns_to_scale = ["age", "annual_income", "spending_score"]

    # Chuyển đổi các cột trong DataFrame
    df_scaled = scale_columns(df, columns_to_scale)

    sse = []
    k_rng = range(1, 11)
    for k in k_rng:
        km = KMeans(n_clusters=k)
        km.fit(df_scaled[["age", "spending_score", "annual_income"]])
        sse.append(km.inertia_)

    km = KMeans(n_clusters=5)
    y_predicted = km.fit_predict(df_scaled[["age", "spending_score", "annual_income"]])
    df_scaled["cluster"] = y_predicted
    df_scaled.head()

    # Get cluster centers (representative elements)
    cluster_centers = km.cluster_centers_

    # Count the number of elements in each cluster
    cluster_counts = df_scaled["cluster"].value_counts().sort_index()

    # Combine the cluster centers and counts into a DataFrame for better visualization
    cluster_info = pd.DataFrame(
        cluster_centers, columns=["age", "spending_score", "annual_income"]
    )
    cluster_info["value"] = cluster_counts.values

    return cluster_info


def calculate_age(day_of_birth):
    # Kiểm tra xem ngày sinh có tồn tại không
    if day_of_birth:
        # Chuyển đổi ngày sinh thành định dạng datetime
        birth_date = datetime.strptime(day_of_birth, "%Y-%m-%d")
        # Lấy ngày hiện tại
        current_date = datetime.now()
        # Tính tuổi
        age = (
            current_date.year
            - birth_date.year
            - (
                (current_date.month, current_date.day)
                < (birth_date.month, birth_date.day)
            )
        )
        return age
    else:
        return None


def scale_columns(df, columns):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled
