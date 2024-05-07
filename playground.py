class PanelUserPreferencessViewSet(viewsets.ModelViewSet):
    queryset = PanelUserPreferences.objects.all()
    serializer_class = PanelUserPreferencessSerializer
    permission_classes = [HasAppToken]

    def get_queryset(self):
        queryset = self.queryset
        user_id = self.request.query_params.get('user_id')

        if user_id:
            user = User.objects.filter(id=user_id)
            if user:
                queryset = queryset.filter(user_id=user)

        return queryset

    def create(self, request, *args, **kwargs):
        # Ensure the user_id is provided in the request
        user_id = request.data.get('user_id')
        if not user_id:
            return Response({"error": "User ID not provided"}, status=400)

        # Attempt to fetch onboarding data for the user
        try:
            onboarding_data = Onboarding.objects.get(user_id=user_id)
        except ObjectDoesNotExist:
            return Response({"error": "Onboarding data not found for this user"}, status=404)

        # Attempt to retrieve the user object
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)

        # Prepare event suggestion data based on the user and their onboarding data
        try:
            event_suggestions_data = self.prepare_event_suggestions_data(user, onboarding_data)
            PanelUserPreferences.objects.bulk_create(event_suggestions_data)
        except Exception as e:
            return Response({"error": f"Failed to create event suggestions: {str(e)}"}, status=400)

        # Return a success response
        return Response({"detail": "Successfully updated"}, status=201)

    def prepare_event_suggestions_data(self, user, onboarding_data):
        """
        Creates the a dictionary and parses and fills out the user's
        onboarding data

        Inputs:
            user: the User object
            onboarding_data (Onboarding): User's rows on Onboarding data from the model

        Returns: event_suggestions_data (dict): dictionary containing user's
        onboarding information as necessary for the PanelUserPreferencess row
        """
        event_suggestions_data = {'user_id': user}
        event_suggestions_data.update(self.parse_num_participants_pref(onboarding_data.num_participants))
        event_suggestions_data.update(self.parse_distance_preferences(onboarding_data.distance))
        event_suggestions_data.update(self.parse_similarity_preferences(onboarding_data.similarity_to_group))
        event_suggestions_data.update(self.parse_similarity_metrics(onboarding_data.similarity_metrics))
        event_suggestions_data.update(self.parse_hobby_type_onboarding(onboarding_data.most_interested_hobby_types.all()))
        event_suggestions_data.update(self.parse_user_availability(user.id))
        return event_suggestions_data

    def parse_user_availability(self, user_id):
        """
        Helper function to parse user availability data, formatting and standardizing
        it into the PanelUserPreferencess row

        Inputs:
            user_id: User's ID

        Returns: availability_data (dict): dictionary of user's availability data
        """
        user_availability = Availability.objects.filter(user_id=user_id)
        time_period_mapping = {
            'early_morning': [5, 6, 7, 8],
            'morning': [9, 10, 11, 12],
            'afternoon': [13, 14, 15, 16],
            'evening': [17, 18, 19, 20],
            'night': [21, 22, 23, 24],
            'late_night': [1, 2, 3, 4]
        }
        availability_data_lst = {}
        for availability in user_availability:
            day_of_week = availability.calendar_id.day_of_week
            hour = availability.calendar_id.hour
            for period, hours in time_period_mapping.items():
                preference_field = f"pref_{day_of_week.lower()}_{period}"
                if int(hour) in hours:
                    availability_data_lst[preference_field] = availability_data_lst.get(preference_field, [])
                    availability_data_lst[preference_field].append(availability.available)
        
        
        availability_data = {key: any(value) for key, value in availability_data_lst.items()}
        return availability_data

    def parse_distance_preferences(self, distance):
        """
        Helper function to parse user distance preferences data, formatting and standardizing
        it into the PanelUserPreferencess row

        Inputs:
            distance (string): The user's distance preferences

        Returns: distance_data (dict): dictionary of user's distance preferences data
        """
        distance_preference_mapping = {
            'Within 1 mile': 'pref_dist_within_1mi',
            'Within 5 miles': 'pref_dist_within_5mi',
            'Within 10 miles': 'pref_dist_within_10mi',
            'Within 15 miles': 'pref_dist_within_15mi',
            'Within 20 miles': 'pref_dist_within_20mi',
            'Within 30 miles': 'pref_dist_within_30mi',
            'Within 40 miles': 'pref_dist_within_40mi',
            'Within 50 miles': 'pref_dist_within_50mi'
        }
        distance_data = {}
        if distance != "No preference":
            distance_data[distance_preference_mapping[distance]] = True
        
        for pref, field in distance_preference_mapping.items():
            if pref != distance:
                distance_data[field] = False

        return distance_data

    def parse_similarity_preferences(self, similarity_value):
        """
        Helper function to parse user similarity preference data, formatting and standardizing
        it into the PanelUserPreferencess row

        Inputs:
            similarity_values (str): a user's similarity preferences data

        Returns: similarity_data (dict): dictionary of user's similarity data
        """

        similarity_mapping = {
            'Completely dissimilar': 'pref_similarity_to_group_1',
            'Moderately dissimilar': 'pref_similarity_to_group_2',
            'Moderately similar': 'pref_similarity_to_group_3',
            'Completely similar': 'pref_similarity_to_group_4',
        }
        similarity_data = {}
        if similarity_value in similarity_mapping:
            similarity_data[similarity_mapping[similarity_value]] = True
        
        for pref, field in similarity_mapping.items():
            if pref != similarity_value:
                similarity_data[field] = False
        
        return similarity_data

    def parse_similarity_metrics(self, metrics):
        """
        Helper function to parse user similarity metrics data, formatting and standardizing
        it into the PanelUserPreferencess row

        Inputs:
            metrics (lst): List of user's similarity metrics data

        Returns: parsed_metrics (dict): dictionary of user's similarity metrics data
        """
        similarity_metrics_mapping = {
            "Gender": "pref_gender_similar",
            "Race or Ethnicity": "pref_race_similar",
            "Age range": "pref_age_similar",
            "Sexual Orientation": "pref_sexual_orientation_similar",
            "Religious Affiliation": "pref_religion_similar",
            "Political Leaning": "pref_political_leaning_similar"
        }

        if not metrics:
            return {metric: False for metric in similarity_metrics_mapping.values()}
        else:
            # If metrics list is not empty, set corresponding values to True
            parsed_metrics = {}
            for preference, field in similarity_metrics_mapping.items():
                parsed_metrics[field] = preference in metrics
            
            return parsed_metrics

    def num_participant_mapping(self, value):
        """
        Helper function to standardize participant data into necessary format
        for EventSugestions Row

        Inputs:
            value (str): string of user's preferred number of event participants, if empty returns Null

       Return: string: String of the value parsed into the necessary format
        """

        if "1-5" in value:
            return "1to5"
        elif "5-10" in value:
            return "5to10"
        elif "10-15" in value:
            return "10to15"
        elif "15+" in value:
            return "15p"
        else:
            return None

    def parse_hobby_type_onboarding(self, hobby_types):
        """
        Helper function to parse user's scenario hobby type data formatting and standardizing
        it into the PanelUserPreferencess row

        Inputs:
            hobby_types (list): list of HobbyType objects

        Returns:
            hobby_data (dict): dictionary of user's scenario hobby type
        """
        hobby_data = {}
        hobby_type_str = [hobby_type.type for hobby_type in hobby_types]

        category_mapping = {
            "TRAVEL": "pref_hobby_category_travel",
            "ARTS AND CULTURE": "pref_hobby_category_arts_and_culture",
            "LITERATURE": "pref_hobby_category_literature",
            "FOOD AND DRINK": "pref_hobby_category_food",
            "COOKING/BAKING": "pref_hobby_category_cooking_and_baking",
            "SPORT/EXERCISE": "pref_hobby_category_exercise",
            "OUTDOORS":  "pref_hobby_category_outdoor_activities",
            "CRAFTING": "pref_hobby_category_crafting",
            "HISTORY AND LEARNING": "pref_hobby_category_history",
            "COMMUNITY EVENTS": "pref_hobby_category_community",
            "GAMING": "pref_hobby_category_gaming",
        }

        for hobby_type, pref_col in category_mapping.items():
            hobby_data[pref_col] = hobby_type in hobby_type_str

        return hobby_data

    def parse_num_participants_pref(self, num_participants):
        """
        Helper function to parse user's preferred number of participants data formatting and standardizing
        it into the PanelUserPreferencess row

        Inputs:
            num_participants (list): The user's preferred number of participants

        Returns:
            num_participants_data (dict): dictionary of user's preferred number of participants data
        """
        assert type(num_participants) == list, "num_participants must be a list"

        num_participants_map = {
            "1-5": "pref_num_particip_1to5",
            "5-10": "pref_num_particip_5to10",
            "10-15": "pref_num_particip_10to15",
            "15+": "pref_num_particip_15p"
        }
        num_participants_data = {}
        for num_participant in num_participants_map:
            num_participants_data[num_participants_map[num_participant]] = num_participant in num_participants

        return(num_participants_data)
